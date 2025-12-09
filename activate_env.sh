#!/bin/bash
# Activation script for the virtual environment

echo "🔧 Activating event-detection virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo ""
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "To deactivate, run: deactivate"

