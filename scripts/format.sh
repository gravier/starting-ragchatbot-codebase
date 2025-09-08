#!/bin/bash
set -e
echo "🔧 Running code formatting..."
echo "📋 Sorting imports with isort..."
uv run isort .
echo "🖤 Formatting code with black..."
uv run black .
echo "✅ Code formatting complete!"