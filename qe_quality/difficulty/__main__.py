"""Run with python -m qe_quality.difficulty; no GPU is used."""

import argparse
from pathlib import Path

from .inventory import prepare
from .io import verify_run
from .pipeline import evaluate, freeze
from .review import contact_sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Index baseline originals and align existing GPU predictions")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("review-sheets", help="Create label-inspection previews in a separate directory")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("evaluate", help="Compute reviewed candidates and evidence-bounded diagnostics")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--reviews", type=Path)
    p.add_argument("--splits", type=Path)
    p.add_argument("--frozen", type=Path, help="Evaluate validation only, with previously frozen rules")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("freeze", help="Freeze development choices before validation")
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("verify", help="Check artifact hashes for an immutable run")
    p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.config, args.output)
        elif args.command == "review-sheets":
            result = contact_sheets(args.prepared, args.output)
        elif args.command == "evaluate":
            result = evaluate(args.prepared, args.output, args.reviews, args.splits, args.frozen)
        elif args.command == "freeze":
            result = freeze(args.development, args.output)
        else:
            verify_run(args.run)
            result = "Artifact hashes verified"
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"difficulty: {exc}\n")
    print(result)


if __name__ == "__main__":
    main()
