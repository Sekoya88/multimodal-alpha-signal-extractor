"""cli.py — Command-Line Interface for the Multimodal Alpha-Signal Extractor.

This serves as the primary terminal entry point, wiring dependencies
via the DI container and running the usecase.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path when running as installed script
# cli.py -> presentation/ -> alpha_signal/ -> src/ -> project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DATASET_DIR
from alpha_signal.infrastructure.logger import setup_logging, logger
from alpha_signal.presentation.di_container import build_analyze_market_usecase


def main() -> None:
    """Parse arguments and execute the Alpha-Signal pipeline via Use Case."""
    parser = argparse.ArgumentParser(
        description="Live Market Analysis with Real Data via Clean Architecture"
    )
    parser.add_argument(
        "--ticker", "-t", type=str, default="AAPL",
        help="Stock ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--days", "-d", type=int, default=60,
        help="Number of trading days for the chart (default: 60)",
    )
    parser.add_argument(
        "--json-logs", action="store_true",
        help="Enable JSON structured logging for production",
    )
    
    args = parser.parse_args()
    
    # Initialize robust logging
    setup_logging(force_json=args.json_logs)

    try:
        logger.info(f"🚀 Initializing Alpha-Signal Extractor for {args.ticker}")
        
        output_dir = DATASET_DIR / "live_sessions"
        
        # 1. Dependency Injection
        usecase = build_analyze_market_usecase(output_dir=output_dir)
        
        # 2. Execute
        decision = asyncio.run(usecase.execute(ticker=args.ticker, days=args.days))
        
        # 3. Present Results
        output_json = decision.model_dump_json(indent=2)
        print(f"\n{output_json}")
        
        output_path = output_dir / f"decision_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        output_path.write_text(output_json, encoding="utf-8")
        logger.info(f"💾 Decision saved → {output_path}")

    except Exception as e:
        logger.exception("A critical error occurred during pipeline execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
