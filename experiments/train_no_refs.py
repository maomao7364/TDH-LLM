"""Reference ablation: remove both daily and weekly references."""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from experiments.reference_ablation_training import make_parser, run_reference_experiment


if __name__ == '__main__':
    parser = make_parser('Reference ablation: no daily or weekly references')
    args = parser.parse_args()
    run_reference_experiment(args, reference_mode='no_refs')
