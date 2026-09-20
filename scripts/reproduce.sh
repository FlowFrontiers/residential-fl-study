#!/usr/bin/env bash
# Run the complete residential matrix, including both capacities and temporal checks.
# Use --dry-run to inspect commands. Set PYTHON to an installed environment's interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" scripts/run_suite.py "$@"
