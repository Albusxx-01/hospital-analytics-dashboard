"""Commands keep data preparation explicit and application startup lightweight."""

import argparse
import logging
from pathlib import Path

from .config import Settings
from .logging_config import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hospital Intelligence · synthetic operational analytics"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "demo"):
        generate_parser = commands.add_parser(name)
        generate_parser.add_argument("--seed", type=int, default=42)
        generate_parser.add_argument("--encounters", type=int, default=50_000)
    commands.add_parser("build")
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--output", type=Path, default=Path("exports"))
    export_parser.add_argument("--hyper", action="store_true")
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    configure_logging()
    settings = Settings(
        root=args.data_dir,
        seed=getattr(args, "seed", 42),
        encounters=getattr(args, "encounters", 50_000),
    )
    try:
        if args.command in {"generate", "demo"}:
            from .generate import generate

            generate(settings)
        if args.command in {"build", "demo"}:
            from .pipeline import build
            from .services import precompute

            build(settings)
            precompute(settings)
        if args.command in {"export", "demo"}:
            from .export import export_data
            from .services import AnalyticsService

            manifest = export_data(
                AnalyticsService(settings.root),
                getattr(args, "output", Path("exports")),
                getattr(args, "hyper", False),
            )
            logging.info("Export complete: %s", manifest["export_rows"])
        if args.command == "serve":
            from .dashboard import create_app

            app = create_app(settings.root)
            app.run(host=args.host, port=args.port, debug=False)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logging.error("%s. Run 'hospital demo' to prepare a complete dataset.", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
