#!/usr/bin/env python
"""Run one compact prompt analysis from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main() -> None:
    """Run one analysis and print compact JSON."""

    from transformer_circuit_visualizer.analysis import CircuitAnalyzer
    from transformer_circuit_visualizer.config import settings
    from transformer_circuit_visualizer.model_service import (
        MockModelService,
        TransformerLensModelService,
    )
    from transformer_circuit_visualizer.schemas import AnalyzeRequest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use TransformerLens instead of the lightweight mock service.",
    )
    args = parser.parse_args()

    if args.real:
        service = TransformerLensModelService()
        model_name = args.model or settings.default_model
    else:
        service = MockModelService()
        model_name = args.model or service.list_models()[0]

    analyzer = CircuitAnalyzer(service)
    result = analyzer.analyze(
        AnalyzeRequest(prompt=args.prompt, model_name=model_name, top_k=args.top_k)
    )
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
