#!/bin/bash
set -e
echo "🚀 Running complete quality check pipeline..."
echo "Step 1/3: Formatting code..."
bash scripts/format.sh
echo "Step 2/3: Running linting..."
bash scripts/lint.sh
echo "Step 3/3: Running tests..."
bash scripts/test.sh
echo "🎉 All quality checks passed! Code is ready for review."