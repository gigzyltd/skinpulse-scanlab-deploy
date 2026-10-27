"""Local Scan Lab command line. It does not call a network or an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scanlab.cv.geometry import build_regions, draw_overlay
from scanlab.cv.images import decode_image, encode_png
from scanlab.cv.pipeline import Settings, analyze_bytes, build_backend
from scanlab.cv.repeatability import load_json, report_from_image_manifest, report_from_measurements
from scanlab.cv.support import ScanLabError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scanlab-cv")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", choices=["geometric", "mediapipe"])
    common.add_argument("--model", default=None)

    analyze = sub.add_parser("analyze", parents=[common])
    analyze.add_argument("image")
    analyze.add_argument("--overlay", default=None)

    quality = sub.add_parser("quality", parents=[common])
    quality.add_argument("image")

    repeat = sub.add_parser("repeatability", parents=[common])
    repeat.add_argument("--images", default=None)
    repeat.add_argument("--measurements", default=None)
    repeat.add_argument("--root", default=None)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--backend", choices=["geometric", "mediapipe"])
    serve.add_argument("--model", default=None)

    args = parser.parse_args(argv)
    settings = _settings(args)
    try:
        if args.command == "analyze":
            return _analyze(args, settings)
        if args.command == "quality":
            return _quality(args, settings)
        if args.command == "repeatability":
            return _repeat(args, settings)
        if args.command == "serve":
            return _serve(args, settings)
    except ScanLabError as exc:
        print(json.dumps({"error": exc.code, "message": exc.message}))
        return 1
    return 1


def _settings(args) -> Settings:
    settings = Settings.from_env()
    backend = getattr(args, "backend", None)
    model = getattr(args, "model", None)
    return settings.overridden(backend=backend, model=model)


def _print(model) -> None:
    print(json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True))


def _analyze(args, settings: Settings) -> int:
    data = Path(args.image).read_bytes()
    result = analyze_bytes(data, settings, include_metrics=True)
    _print(result)
    if args.overlay and result.landmark_backend:
        _write_overlay(data, settings, args.overlay)
    return 0 if result.accepted else 2


def _quality(args, settings: Settings) -> int:
    data = Path(args.image).read_bytes()
    result = analyze_bytes(data, settings, include_metrics=False)
    _print(result)
    return 0


def _write_overlay(data: bytes, settings: Settings, destination: str) -> None:
    rgb = decode_image(data, max_edge_px=settings.max_edge_px, max_bytes=settings.max_bytes)
    detection = build_backend(settings).detect(rgb)
    if detection.anchors is None:
        return
    regions = build_regions(detection.anchors)
    overlay = draw_overlay(rgb, detection.anchors, regions)
    Path(destination).write_bytes(encode_png(overlay))


def _repeat(args, settings: Settings) -> int:
    if bool(args.images) == bool(args.measurements):
        raise ScanLabError("invalid_usage", "Pass exactly one of --images or --measurements.")
    if args.measurements:
        report = report_from_measurements(load_json(Path(args.measurements)))
    else:
        root = Path(args.root).resolve() if args.root else None
        report = report_from_image_manifest(load_json(Path(args.images)), settings, root=root)
    _print(report)
    return 0


def _serve(args, settings: Settings) -> int:
    import uvicorn

    from scanlab.cv.api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
