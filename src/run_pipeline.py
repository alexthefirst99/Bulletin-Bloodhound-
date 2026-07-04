"""End-to-end: build index -> retrieve -> generate -> evaluate, for one mode."""
import argparse

from build_index import build
from evaluate_generation import evaluate as eval_gen
from evaluate_retrieval import evaluate as eval_ret
from generate import run as run_generate
from retrieve import run as run_retrieve


def main(mode):
    print(f"\n=== {mode.upper()} :: build index ===")
    build(mode)
    print(f"\n=== {mode.upper()} :: retrieve ===")
    run_retrieve(mode)
    print(f"\n=== {mode.upper()} :: retrieval metrics ===")
    eval_ret(mode)
    print(f"\n=== {mode.upper()} :: generate ===")
    run_generate(mode)
    print(f"\n=== {mode.upper()} :: generation metrics ===")
    eval_gen(mode)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "engineered"], required=True)
    args = p.parse_args()
    main(args.mode)
