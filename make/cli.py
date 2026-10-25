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


def _texture(args: argparse.Namespace) -> int:
    from make import settings
    from make.comfyui import availability
    from make.generation import ComfyUIClient
    from make.texturing.pipeline import texture_single_view
    from make.texturing.testmesh import build_test_mesh

    availability.check()
    stem = args.mesh.stem if args.mesh else "test_mesh"
    work = args.work_dir or settings.work_dir() / "texturing" / f"{stem}_seed{args.seed}"
    mesh = args.mesh or build_test_mesh(work / "test_mesh.blend")
    out = args.out or work / f"{stem}_textured.blend"
    result = texture_single_view(mesh_blend=mesh, reference=args.reference, prompt=args.prompt,
                                 seed=args.seed, out_blend=out, client=ComfyUIClient(), work_dir=work)
    for step, seconds in result.timings.items():
        print(f"  {step:<18} {seconds:6.2f}s")
    print(f"blend:   {result.blend}\npreview: {result.preview}")
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

    tex = sub.add_parser("texture", help="texture a mesh from one front view (spike #18)")
    tex.add_argument("--mesh", type=Path, help=".blend with one UV-mapped mesh (default: generated test mesh)")
    tex.add_argument("--reference", type=Path, required=True)
    tex.add_argument("--prompt", required=True)
    tex.add_argument("--seed", type=int, default=0)
    tex.add_argument("--out", type=Path, help="output .blend (default: in the work dir)")
    tex.add_argument("--work-dir", type=Path, help="intermediates (default: ACC_WORK_DIR/texturing/<mesh>_seed<n>)")
    tex.set_defaults(func=_texture)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
