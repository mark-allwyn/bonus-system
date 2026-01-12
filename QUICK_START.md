# Quick Start Guide

## 1. Install Dependencies (30 seconds)

```bash
pip install -r requirements.txt
```

## 2. Generate All Results (2-5 minutes)

```bash
python src/generate_results.py
```

This creates:
- Synthetic data (10,000 players)
- Trained models (uplift, LTV, churn)
- 4 publication figures (PNG, 300 DPI)
- 3 results tables (CSV + LaTeX)
- Summary report

## 3. View Results

```bash
# Read summary
cat results/RESULTS_SUMMARY.txt

# View figures
open results/figures/

# Check tables
open results/tables/
```

## 4. Explore Interactively

```bash
jupyter notebook notebooks/
```

Run notebooks in order:
1. Data Generation
2. Model Training
3. Results Analysis

## 5. Modify Parameters

Edit `configs/simulation_config.yaml`:

```yaml
population:
  total_size: 20000        # Scale up to 20k players
  segments:
    persuadables: 0.25     # Change segment proportions

treatment_effects:
  persuadables: 0.80       # Adjust treatment effects
```

Then regenerate:
```bash
python src/generate_results.py
```

## Key Outputs for Paper

### Figures (results/figures/)
- `figure1_segment_distribution.png` - Four player segments
- `figure2_uplift_calibration.png` - Model calibration
- `figure3_cumulative_gains.png` - Performance comparison
- `figure4_feature_importance.png` - Key features

### Tables (results/tables/)
- `table1_model_comparison.tex` - Performance metrics
- `table2_segment_analysis.tex` - Segment characteristics
- `summary_statistics.csv` - Key numbers

### Key Statistics

Check `results/RESULTS_SUMMARY.txt` for:
- % improvement over random targeting
- Uplift vs response model advantage
- Segment distributions
- Treatment effects by segment

## Common Commands

```bash
# Just generate data
python src/data_generator.py

# Just train uplift model
python src/uplift_model.py

# Just run evaluation
python src/evaluation.py

# Full pipeline (recommended)
python src/generate_results.py
```

## File Structure

```
├── src/                    # Python modules
├── notebooks/              # Jupyter notebooks
├── configs/                # simulation_config.yaml
├── data/                   # Generated datasets & models
├── results/
│   ├── figures/           # PNG figures (300 DPI)
│   └── tables/            # CSV + LaTeX tables
├── README.md              # Full documentation
├── CLAUDE.md              # Developer guide
└── PROJECT_SUMMARY.md     # What was built
```

## Need Help?

- **Full docs**: See `README.md`
- **Dev guide**: See `CLAUDE.md`
- **Overview**: See `PROJECT_SUMMARY.md`

## One-Line Setup (Alternative)

```bash
./run_pipeline.sh
```

This script:
1. Creates virtual environment
2. Installs dependencies
3. Runs complete pipeline
4. Shows output locations

---

**That's it!** You now have all results needed for your paper.
