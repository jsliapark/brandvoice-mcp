"""Entry point for `python -m brandvoice_mcp`."""

import sys


def main() -> None:
    """Launch the brandvoice-mcp server."""
    try:
        from brandvoice_mcp.server import run_server

        run_server()
    except KeyboardInterrupt:
        pass
    except EnvironmentError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
