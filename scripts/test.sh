#!/bin/bash
set -e
echo "🧪 Running tests..."
echo "🐍 Running pytest..."
uv run pytest backend/tests/ -v
echo "✅ Tests complete!"