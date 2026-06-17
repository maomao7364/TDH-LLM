"""Ablation: replace GPT-2 with randomly-initialised TransformerEncoder."""
import sys, os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from experiments._training import make_parser, seed_it, run_experiment

if __name__ == '__main__':
    args = make_parser('DeST-LLM w/o LLM (random Transformer)').parse_args()
    seed_it()
    run_experiment(args, variant='no_llm', backbone_type='gpt2')
