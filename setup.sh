#!/bin/bash
set -e

echo "========================================"
echo "  MedGuard Setup"
echo "  Uncertainty-Aware Medical Image"
echo "  Classification Pipeline"
echo "========================================"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/3] Virtual environment already exists."
fi

echo "[2/3] Installing dependencies..."
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[3/3] Creating directories..."
mkdir -p data models results

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Run the pipeline:"
echo "  python run.py --phase 1          # Train baseline + DenseNet121"
echo "  python run.py --phase 2          # + Hybrid GMM & OOD detection"
echo "  python run.py --phase 3          # + Diagnostic analysis & Grad-CAM"
echo ""
echo "Demo mode (uses saved models, no training):"
echo "  python run.py --phase 2 --demo"
echo ""
echo "Launch the dashboard:"
echo "  streamlit run app.py"
