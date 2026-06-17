import gc
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


def _create_random_gpt2(gpt_layers, dropout=0.15):
    """Fallback: create a randomly-initialised GPT-2 when no pretrained weights are available."""
    config = GPT2Config(
        n_embd=768,
        n_head=12,
        n_layer=gpt_layers,
        n_positions=1024,
        n_inner=3072,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
        embd_pdrop=dropout,
    )
    return GPT2Model(config)


class PartiallyFrozenAttention(nn.Module):
    """
    GPT-2 with partially frozen layers.
    - Bottom (L - U) layers: freeze everything except LayerNorm
    - Top U layers: fully unfrozen (attention + MLP + LayerNorm)
    Adapted from ST-LLM's PFA strategy.
    """

    def __init__(self, gpt_layers=6, U=1, pretrained=True, dropout=0.15,
                 gpt2_path=None):
        super().__init__()

        if pretrained:
            model_id = gpt2_path if gpt2_path else "gpt2"
            try:
                self.gpt2 = GPT2Model.from_pretrained(
                    model_id,
                    local_files_only=True,
                )
                print(f"[PFA] Loaded GPT-2 from {model_id}")
            except (OSError, IOError, EnvironmentError):
                if gpt2_path:
                    raise FileNotFoundError(
                        f"GPT-2 not found at {gpt2_path}. "
                        f"Download it first or check the path."
                    )
                try:
                    self.gpt2 = GPT2Model.from_pretrained("gpt2")
                except (OSError, IOError, EnvironmentError):
                    print("[PFA] Cannot load GPT-2 weights, using random init")
                    self.gpt2 = _create_random_gpt2(gpt_layers, dropout)
        else:
            self.gpt2 = _create_random_gpt2(gpt_layers, dropout)

        # Override dropout in pretrained config
        self.gpt2.config.attn_pdrop = dropout
        self.gpt2.config.resid_pdrop = dropout
        self.gpt2.config.embd_pdrop = dropout
        for block in self.gpt2.h:
            block.attn.attn_dropout.p = dropout
            block.attn.resid_dropout.p = dropout
            block.mlp.dropout.p = dropout

        self.gpt2.h = self.gpt2.h[:gpt_layers]
        # Strip unused token embedding — we always pass inputs_embeds,
        # so the 38.6M-param wte (50257 × 768) is never used.
        if hasattr(self.gpt2, 'wte'):
            self.gpt2.wte = nn.Embedding(1, self.gpt2.wte.embedding_dim)

        # Force cleanup of orphaned full-model layers
        gc.collect()

        self.U = U

        for layer_idx, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_idx < gpt_layers - U:
                    if 'ln' in name or 'wpe' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state
