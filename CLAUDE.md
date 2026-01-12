# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase supporting an academic paper on **uplift modeling for player bonus allocation** in gaming. The repository contains:
- Synthetic data generation with four player segments (Persuadables, Sure Things, Lost Causes, Sleeping Dogs)
- T-Learner implementation for causal inference
- Supporting models (LTV, Churn) and composite scoring
- Publication-ready evaluation framework

**Key Goal**: Validate theoretical framework and generate empirical results (figures/tables) that can be added to the paper.

## Development Commands

### Complete Pipeline

Generate all results from scratch:
```bash
cd src
python generate_results.py
```

This runs the full pipeline: data generation → model training → evaluation → figure/table generation.

### Individual Components

Run specific modules:
```bash
# Generate synthetic data only
python src/data_generator.py

# Train uplift model only
python src/uplift_model.py

# Train LTV and churn models
python src/ltv_model.py
python src/churn_model.py

# Generate evaluation metrics
python src/evaluation.py

# Test scoring pipeline
python src/scoring.py
```

### Interactive Analysis

Launch Jupyter notebooks:
```bash
jupyter notebook notebooks/
```

Three notebooks:
1. `01_data_generation.ipynb` - Create and explore synthetic data
2. `02_model_training.ipynb` - Train all models
3. `03_results_analysis.ipynb` - Generate publication figures/tables

### Testing

Test individual modules:
```bash
# Each module has __main__ block for testing
python src/features.py
python src/utils.py
```

## Architecture

### Data Flow

```
1. Config (simulation_config.yaml)
   ↓
2. Data Generator → player_data.csv (10k players, 4 segments)
   ↓
3. Feature Engineering → Scaled feature matrix
   ↓
4. Model Training:
   - Uplift Model (T-Learner) → uplift_score
   - LTV Model → ltv_score
   - Churn Model → churn_probability
   ↓
5. Scoring Pipeline → composite_score, player selection
   ↓
6. Evaluation → Figures (PNG) + Tables (CSV/LaTeX)
```

### Three-Model System

**Core Architecture** (from paper):
1. **Uplift Model**: Estimates causal treatment effect using T-Learner
2. **LTV Model**: Predicts player lifetime value
3. **Churn Model**: Predicts churn probability

**Composite Score Formula**:
```
Priority = Uplift × LTV × Churn_Probability × Business_Rules
```

### T-Learner Implementation

The uplift model (`src/uplift_model.py`) implements the T-Learner approach:
- Trains **two separate Random Forest models**:
  - `treatment_model`: Trained on players who received bonuses
  - `control_model`: Trained on players who did not receive bonuses
- **Uplift prediction**: `P(Y|X,T=1) - P(Y|X,T=0)`
- Captures heterogeneous treatment effects across player segments

### Four Player Segments

Defined in `src/data_generator.py`, controlled by `configs/simulation_config.yaml`:

| Segment | Proportion | Treatment Effect | Targeting Decision |
|---------|-----------|------------------|-------------------|
| Persuadables | 20% | +0.70 | **TARGET** (positive ROI) |
| Sure Things | 40% | 0.00 | SKIP (wasted spend) |
| Lost Causes | 35% | 0.00 | SKIP (no effect) |
| Sleeping Dogs | 5% | -0.40 | **AVOID** (negative effect) |

**Critical Insight**: Traditional response models fail by targeting Sure Things (high response but no uplift) and Sleeping Dogs (negative uplift). Uplift models correctly identify only Persuadables.

## Key Files

### Configuration
- `configs/simulation_config.yaml` - **Single source of truth** for all parameters
  - Segment proportions and treatment effects
  - Feature distributions by segment
  - Model hyperparameters
  - Business rules (budget constraints, thresholds)

### Core Modules
- `src/data_generator.py` - Generates realistic synthetic players with segment structure
- `src/features.py` - Feature engineering with 9 core features (monetary, behavioral, temporal)
- `src/uplift_model.py` - T-Learner + baseline ResponseModel for comparison
- `src/scoring.py` - Composite scoring and player selection logic
- `src/evaluation.py` - Uplift-specific metrics (AUUC, Qini curves, calibration)

### Master Script
- `src/generate_results.py` - **Run this to regenerate all paper results**

## Modifying Parameters

### Change Segment Distribution

Edit `configs/simulation_config.yaml`:
```yaml
population:
  segments:
    persuadables: 0.25    # Increase to 25%
    sleeping_dogs: 0.03   # Decrease to 3%
```

### Adjust Treatment Effects

```yaml
treatment_effects:
  persuadables: 0.80      # Stronger positive effect
  sleeping_dogs: -0.50    # Stronger negative effect
```

### Scale Population Size

```yaml
population:
  total_size: 20000       # Increase to 20k players
```

### Tune Model Hyperparameters

```yaml
models:
  uplift:
    n_estimators: 200     # More trees
    max_depth: 15         # Deeper trees
```

**After changes**: Run `python src/generate_results.py` to regenerate all results.

## Common Development Tasks

### Add New Feature

1. Edit `configs/simulation_config.yaml` to define feature distribution per segment
2. Add feature generation logic in `src/data_generator.py` → `_generate_features()`
3. Add feature name to `src/features.py` → `FEATURE_COLUMNS`
4. Regenerate data: `python src/data_generator.py`

### Implement Alternative Uplift Model

1. Create new class in `src/uplift_model.py` (e.g., `SLearner`, `XLearner`)
2. Implement `fit()` and `predict_uplift()` methods
3. Update `src/generate_results.py` to use new model
4. Compare results with T-Learner in evaluation

### Add New Evaluation Metric

1. Add method to `src/evaluation.py` → `UpliftEvaluator` class
2. Call from `src/generate_results.py` or notebooks
3. Example: `def calculate_qini_coefficient()`

### Generate Custom Figures

Copy pattern from `src/evaluation.py`:
```python
def plot_custom_analysis(self, df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
    # Your plotting code
    if save_path:
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
    return fig
```

## Output Locations

All generated outputs:

### Data
- `data/player_data.csv` - Full synthetic dataset
- `data/player_data_train.csv` - Training set (70%)
- `data/player_data_test.csv` - Test set (30%)
- `data/*.pkl` - Serialized models

### Results
- `results/figures/*.png` - High-resolution figures (300 DPI)
- `results/tables/*.csv` - Results tables
- `results/tables/*.tex` - LaTeX-formatted tables
- `results/RESULTS_SUMMARY.txt` - Text summary of findings

## Model Files

Trained models are saved as pickle files:
- `data/uplift_model.pkl` - T-Learner uplift model
- `data/response_model.pkl` - Baseline response model
- `data/ltv_model.pkl` - Lifetime value model
- `data/churn_model.pkl` - Churn prediction model

Load models:
```python
from src.uplift_model import TLearner
uplift_model = TLearner.load('data/uplift_model.pkl')
```

## Troubleshooting

### Issue: Low uplift model performance
- Check segment separation in data: `python src/data_generator.py`
- Verify treatment effects in config are sufficiently different
- Increase sample size if segments are small

### Issue: Figures not generating
- Ensure `results/figures/` directory exists: `mkdir -p results/figures`
- Check matplotlib backend if running headless
- Verify DPI setting in `configs/simulation_config.yaml`

### Issue: Model training fails
- Check for missing values: Use `src/utils.py` validation
- Verify feature engineering: `python src/features.py`
- Ensure train/test split maintains segment balance

## Paper Integration

### Updating Paper with Results

After running `python src/generate_results.py`:

1. **Figures**: Copy from `results/figures/` to paper directory
   - All figures are 300 DPI PNG, publication-ready
   - Consider converting to PDF for LaTeX: `convert figure1.png figure1.pdf`

2. **Tables**: Use LaTeX files directly
   - `\input{results/tables/table1_model_comparison.tex}`

3. **Statistics**: Check `results/RESULTS_SUMMARY.txt` for key numbers
   - Update claims in paper (e.g., "X% improvement over random")

### Reproducibility

The entire pipeline is deterministic (fixed random seeds):
```python
random_seed: 42  # In simulation_config.yaml
```

Anyone can reproduce exact results by:
1. Clone repository
2. `pip install -r requirements.txt`
3. `python src/generate_results.py`

## Dependencies

Core libraries (see `requirements.txt`):
- **scikit-learn** - T-Learner, base models
- **pandas, numpy** - Data manipulation
- **matplotlib, seaborn** - Visualizations
- **xgboost** - Optional, higher performance models
- **scipy** - Statistical tests
- **pyyaml** - Configuration

Install all: `pip install -r requirements.txt`

## Important Notes

- **Synthetic data only**: This is a proof-of-concept. All data is generated, not real players.
- **Research purpose**: Designed to validate theoretical framework, not production deployment.
- **Parameterizable**: Easy to experiment with different scenarios via config file.
- **Publication focus**: Primary goal is generating figures/tables for the paper.

## Next Steps for Development

1. **Sensitivity analysis**: Test various parameter combinations
2. **Alternative models**: Implement S-Learner, X-Learner, Causal Forests
3. **Additional features**: Add time-series patterns, seasonality
4. **Larger scale**: Test with 100k+ players
5. **Statistical tests**: Add confidence intervals, significance tests
