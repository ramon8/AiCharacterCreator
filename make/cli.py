"""`make` command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _comfyui(args: argparse.Namespace) -> int:
    from make.comfyui import availability, install, server

    if args.action == "install":
        install.install()
    elif args.action == "start":
        server.start()
    elif args.action == "stop":
        server.stop()
    elif args.action == "check":
        try:
            availability.check()
        except availability.ComfyUIUnavailable as error:
            print(error, file=sys.stderr)
            return 1
        print("ComfyUI OK: server up, all nodes and models present")
    return 0


def _generate(args: argparse.Namespace) -> int:
    from make.comfyui import availability
    from make.generation import ComfyUIClient, Conditions

    availability.check()
    conditions = Conditions.from_files(
        depth=args.depth, normal=args.normal, prompt=args.prompt, seed=args.seed,
        references=tuple(args.reference), mask=args.mask,
    )
    image = ComfyUIClient().generate(conditions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make")
    sub = parser.add_subparsers(dest="command", required=True)

    comfy = sub.add_parser("comfyui", help="manage the local headless ComfyUI")
    comfy.add_argument("action", choices=["install", "start", "stop", "check"])
    comfy.set_defaults(func=_comfyui)

    gen = sub.add_parser("generate", help="generate one image from depth/normal + reference")
    gen.add_argument("--depth", type=Path, required=True)
    gen.add_argument("--normal", type=Path, required=True)
    gen.add_argument("--reference", type=Path, action="append", default=[])
    gen.add_argument("--mask", type=Path)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", type=Path, required=True)
    gen.set_defaults(func=_generate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
