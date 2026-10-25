"""Launch the optional, loopback-only Gradio UI against a configured API origin."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    from pvdiag.ui.client import ApiClient

    try:
        ApiClient(args.api_url)
        if not 1 <= args.port <= 65535:
            raise ValueError("invalid port")
    except ValueError:
        parser.exit(2, "invalid UI configuration; use a loopback HTTP or HTTPS API origin\n")
    try:
        from pvdiag.ui.app import CSS, build_app, gr
    except ImportError:
        parser.exit(2, "install requirements/ui-macos-arm64.txt in a separate UI environment\n")
    build_app(args.api_url).queue(max_size=32).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        show_error=False,
        run_history=False,
        footer_links=[],
        enable_monitoring=False,
        blocked_paths=[str(Path.cwd())],
        allowed_paths=[],
        css=CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=["system-ui"],
            font_mono=["monospace"],
        ),
    )


if __name__ == "__main__":
    main()
