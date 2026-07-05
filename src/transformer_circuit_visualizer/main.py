"""Application entrypoint."""

from __future__ import annotations

import uvicorn

from transformer_circuit_visualizer.api import create_app


app = create_app()


def run() -> None:
    """Run the API server with uvicorn."""

    uvicorn.run("transformer_circuit_visualizer.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
