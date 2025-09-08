#!/bin/bash
set -e
echo "🔍 Running code quality checks..."
echo "🐍 Running flake8 linting..."
uv run flake8 backend/
echo "🔍 Running mypy type checking..."
uv run mypy backend/
echo "✅ Code quality checks complete!"