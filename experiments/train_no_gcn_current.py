"""Module ablation: remove static GCN spatial propagation.

Alias of no_spatial, kept for clearer experiment naming.
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from experiments.module_ablation_training import make_parser, run_module_experiment


if __name__ == '__main__':
    parser = make_parser('Module ablation: no GCN')
    args = parser.parse_args()
    run_module_experiment(args, module_variant='no_spatial')
