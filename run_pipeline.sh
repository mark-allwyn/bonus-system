#!/bin/bash

# Complete Pipeline Runner for Uplift Modeling Research
# Generates all data, trains models, and produces publication-ready results

echo "=========================================="
echo "Uplift Modeling Simulation Pipeline"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "=========================================="
echo "Running Complete Pipeline"
echo "=========================================="
echo ""

# Run the master script
cd src
python generate_results.py

cd ..

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - Figures: results/figures/"
echo "  - Tables: results/tables/"
echo "  - Summary: results/RESULTS_SUMMARY.txt"
echo ""
echo "Next steps:"
echo "  - View results: cat results/RESULTS_SUMMARY.txt"
echo "  - Explore notebooks: jupyter notebook notebooks/"
echo "  - Open figures: open results/figures/"
echo ""
