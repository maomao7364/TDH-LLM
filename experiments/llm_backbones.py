"""
Unified LLM backbone wrapper with PFA (Partially Frozen Attention) support.

Supports: GPT-2, GPT-2-random, Qwen2.5-0.5B, LLaMA (TinyLlama / Llama-3.2-1B).

Each backbone:
  - loads a pretrained decoder-only transformer
  - keeps only the first `num_layers` decoder layers
  - applies PFA: bottom layers freeze all except norms; top U layers unfreeze
    attention, freeze MLP
  - accepts `inputs_embeds` (bypassing token embedding) and returns
    `last_hidden_state`
"""
import sys
import os

# Ensure the DeST-LLM/ package directory is on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)  # DeST-LLM/
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import torch
import torch.nn as nn
import warnings

from transformers import AutoModel, AutoConfig


# ---------------------------------------------------------------------------
# Per-architecture metadata
# ---------------------------------------------------------------------------
ARCH_META = {
    # GPT-2 family
    'gpt2': {
        'model_id': 'gpt2',
        'layers_attr': 'h',                # attribute name for the layer list
        'embed_attr': 'wte',               # token embedding attribute (unused with inputs_embeds)
        'norm_keywords': ['ln'],            # param name keywords that indicate layer-norm
        'attn_keywords': ['attn', 'c_attn', 'c_proj'],
        'mlp_keywords': ['mlp', 'c_fc'],
        'hidden_dim': 768,
    },
    'gpt2-random': {
        'model_id': None,                   # special: random init from config
        'layers_attr': 'h',
        'embed_attr': 'wte',
        'norm_keywords': ['ln'],
        'attn_keywords': ['attn', 'c_attn', 'c_proj'],
        'mlp_keywords': ['mlp', 'c_fc'],
        'hidden_dim': 768,
    },
    # Qwen2 family
    'qwen2': {
        'model_id': 'Qwen/Qwen2.5-0.5B',
        'layers_attr': 'model.layers',       # dot-separated path
        'embed_attr': 'embed_tokens',
        'norm_keywords': ['input_layernorm', 'post_attention_layernorm', 'norm'],
        'attn_keywords': ['self_attn', 'attn'],
        'mlp_keywords': ['mlp'],
        'hidden_dim': 896,
    },
    # LLaMA family (uses TinyLlama as accessible alternative)
    'llama': {
        'model_id': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'layers_attr': 'model.layers',
        'embed_attr': 'embed_tokens',
        'norm_keywords': ['input_layernorm', 'post_attention_layernorm', 'norm'],
        'attn_keywords': ['self_attn', 'attn'],
        'mlp_keywords': ['mlp'],
        'hidden_dim': 2048,
    },
}


def _resolve_attr(obj, path):
    """Resolve a dot-separated attribute path on *obj*."""
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


def _matches_any(name, keywords):
    """Return True if *name* contains any of the *keywords*."""
    return any(kw in name for kw in keywords)


# ---------------------------------------------------------------------------
# Model creation helpers
# ---------------------------------------------------------------------------
def _create_random_gpt2(num_layers, hidden_dim=768):
    """Create a randomly-initialised GPT-2 with a given number of layers."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config

    config = GPT2Config(
        n_embd=hidden_dim,
        n_head=12,
        n_layer=num_layers,
        n_positions=1024,
        n_inner=hidden_dim * 4,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
    )
    return GPT2Model(config)


def _load_pretrained(model_id, num_layers, offline_first=True):
    """Load a pretrained HuggingFace model, keeping only *num_layers* decoder layers."""
    if offline_first:
        try:
            model = AutoModel.from_pretrained(
                model_id, output_hidden_states=True,
                local_files_only=True,
            )
        except (OSError, IOError, EnvironmentError):
            model = AutoModel.from_pretrained(
                model_id, output_hidden_states=True,
            )
    else:
        model = AutoModel.from_pretrained(
            model_id, output_hidden_states=True,
        )

    return model


def _slice_layers(model, meta, num_layers):
    """Keep only the first *num_layers* decoder layers."""
    layers = _resolve_attr(model, meta['layers_attr'])
    trimmed = layers[:num_layers]
    # Write back
    parts = meta['layers_attr'].split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], trimmed)
    return model


def _get_layer_list(model, meta):
    """Return the ModuleList / list of decoder layers."""
    return _resolve_attr(model, meta['layers_attr'])


def _strip_embedding(model, meta):
    """Replace the token embedding with a tiny placeholder.

    We always pass ``inputs_embeds`` so the full embedding matrix
    (e.g. GPT-2's 38.6M-param wte, Qwen2.5's 136M-param embed_tokens)
    is never used. Replacing it frees significant GPU memory.
    """
    embed_attr = meta.get('embed_attr')
    if embed_attr and hasattr(model, embed_attr):
        old = getattr(model, embed_attr)
        setattr(model, embed_attr, nn.Embedding(1, old.embedding_dim))


# ---------------------------------------------------------------------------
# PFA freezing logic
# ---------------------------------------------------------------------------
def _apply_pfa(model, meta, num_layers, U):
    """
    Partially Frozen Attention strategy.

    - Bottom (num_layers - U) layers: freeze everything except norms.
    - Top U layers: fully unfrozen.
    - Embedding / head parameters are frozen (we use inputs_embeds anyway).
    """
    layers = _get_layer_list(model, meta)
    L = len(layers)  # may differ from num_layers if slicing changed it
    if L == 0:
        return

    # Also freeze embedding parameters and final norm
    for name, param in model.named_parameters():
        # Skip decoder layers — handled separately below
        if meta['layers_attr'].split('.')[0] in name:
            continue
        param.requires_grad = False

    for layer_idx, layer in enumerate(layers):
        is_top = layer_idx >= (L - U)

        for name, param in layer.named_parameters():
            if is_top:
                # Top U layers: fully unfrozen
                param.requires_grad = True
            else:
                # Bottom layers: freeze everything except norms
                if _matches_any(name, meta['norm_keywords']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False


# ---------------------------------------------------------------------------
# Main backbone class
# ---------------------------------------------------------------------------
class LLMBackbone(nn.Module):
    """
    Unified LLM backbone with PFA.

    Parameters
    ----------
    backbone_type : str
        One of 'gpt2', 'gpt2-random', 'qwen2', 'llama'.
    num_layers : int
        Number of decoder layers to keep (from the bottom).
    U : int
        Number of top layers whose attention is unfrozen.
    """

    def __init__(self, backbone_type='gpt2', num_layers=3, U=1, gpt2_path=None,
                 dropout=0.15):
        super().__init__()
        assert backbone_type in ARCH_META, f'Unknown backbone: {backbone_type}'
        self.backbone_type = backbone_type
        self.meta = meta = ARCH_META[backbone_type]
        self.num_layers = num_layers
        self.U = U
        self.hidden_dim = meta['hidden_dim']

        # --- load or create model ---
        model_id = gpt2_path if gpt2_path else meta['model_id']
        if backbone_type == 'gpt2-random':
            model = _create_random_gpt2(num_layers, self.hidden_dim, dropout)
        else:
            model = _load_pretrained(model_id, num_layers, offline_first=True)
            model = _slice_layers(model, meta, num_layers)
            # Apply dropout to pretrained config
            if hasattr(model.config, 'attn_pdrop'):
                model.config.attn_pdrop = dropout
                model.config.resid_pdrop = dropout
                model.config.embd_pdrop = dropout

        # Strip unused token embedding before applying PFA
        _strip_embedding(model, meta)

        self.model = model

        # --- apply PFA ---
        _apply_pfa(self.model, meta, num_layers, U)

        # --- count trainable parameters ---
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f'[LLMBackbone] {backbone_type} | layers={num_layers} U={U} | '
              f'trainable={trainable:,}/{total:,} ({100*trainable/total:.1f}%)')

    def forward(self, inputs_embeds):
        """
        Parameters
        ----------
        inputs_embeds : Tensor [B, N, hidden_dim]
            Pre-computed embeddings (bypasses token embedding).

        Returns
        -------
        last_hidden_state : Tensor [B, N, hidden_dim]
        """
        return self.model(inputs_embeds=inputs_embeds).last_hidden_state


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------
_BACKBONE_CACHE = {}

def create_backbone(backbone_type, num_layers=3, U=1, gpt2_path=None,
                    dropout=0.15, clear_cache=False):
    """
    Factory: create (or return cached) LLMBackbone.
    """
    cache_key = (backbone_type, num_layers, U, gpt2_path, dropout)
    if clear_cache:
        _BACKBONE_CACHE.clear()
    if cache_key not in _BACKBONE_CACHE:
        _BACKBONE_CACHE[cache_key] = LLMBackbone(backbone_type, num_layers, U,
                                                 gpt2_path=gpt2_path,
                                                 dropout=dropout)
    return _BACKBONE_CACHE[cache_key]


def get_backbone_hidden_dim(backbone_type):
    """Return the hidden dimension for a backbone type without creating it."""
    return ARCH_META[backbone_type]['hidden_dim']


