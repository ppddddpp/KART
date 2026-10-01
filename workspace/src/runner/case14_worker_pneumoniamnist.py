import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from experiment_case.case14_generalization import run_case14

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-hours", type=float, default=None)
    args = parser.parse_args()
    run_case14(dataset_name="pneumoniamnist", suite=args.suite, max_hours=args.max_hours)
