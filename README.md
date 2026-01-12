# Uplift Modeling for Player Bonus Allocation

This repository contains the complete implementation and simulation code supporting the academic paper **"A Machine Learning Framework for Optimising Player Bonus Allocation Using Causal Inference and Uplift Modelling"** by Mark Stent (January 2026).

## Overview

The paper proposes a comprehensive framework for optimizing player bonus allocation in the gaming industry by identifying which players will respond **because of** bonuses (causal effect), not just who will respond. This implementation demonstrates the theoretical framework with controlled synthetic data and generates publication-ready results.

### Four Player Segments

The framework identifies four distinct behavioral segments based on treatment response:

1. **Persuadables** (20%) - Engage only with bonus (TARGET - positive ROI)
2. **Sure Things** (40%) - Engage regardless of bonus (SKIP - wasted spend)
3. **Lost Causes** (35%) - Never engage (SKIP - no effect)
4. **Sleeping Dogs** (5%) - Disengage with bonus (AVOID - negative effect)

## Repository Contents

### Academic Paper

- `ml-bonus-system-paper-with-simulation.docx` - Updated paper with simulation results integrated
- `ml-bonus-system-paper-final (10).docx` - Original theoretical paper
- `PAPER_UPDATE_SUMMARY.md` - Detailed explanation of changes

### Implementation Code

```
├── src/                          # Python implementation (9 modules)
│   ├── data_generator.py         # Synthetic player data generation
│   ├── features.py               # Feature engineering pipeline
│   ├── uplift_model.py           # T-Learner uplift model + Response model
│   ├── ltv_model.py              # Lifetime value prediction
│   ├── churn_model.py            # Churn probability prediction
│   ├── scoring.py                # Composite scoring pipeline
│   ├── evaluation.py             # Uplift metrics and visualizations
│   ├── utils.py                  # Helper functions
│   └── generate_results.py       # Master pipeline script
│
├── notebooks/                    # Jupyter notebooks (3 analyses)
│   ├── 01_data_generation.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
│
├── configs/
│   └── simulation_config.yaml    # All simulation parameters
│
├── data/                         # Generated datasets and models
├── results/
│   ├── figures/                  # Publication figures (PNG, 300 DPI)
│   └── tables/                   # Results tables (CSV + LaTeX)
│
└── Documentation
    ├── README.md                 # This file
    ├── CLAUDE.md                 # Developer guide
    ├── QUICK_START.md            # 5-minute guide
    ├── PROJECT_SUMMARY.md        # Architecture overview
    ├── DELIVERABLES.md           # Complete deliverables list
    └── RESULTS_FOR_PAPER.md      # Paper-ready content
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Generate All Results

Run the complete pipeline (2-5 minutes):

```bash
python src/generate_results.py
```

This will:
- Generate 10,000 synthetic players with 4 behavioral segments
- Train uplift, LTV, and churn models
- Generate predictions and composite scores
- Create 4 publication-quality figures (300 DPI)
- Export 3 formatted tables (CSV + LaTeX)
- Produce summary report

### 3. Interactive Analysis

Explore step-by-step with Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

Run the notebooks in order:
1. Data Generation - Create and explore synthetic data
2. Model Training - Train all models and view predictions
3. Results Analysis - Generate publication figures and tables

## Simulation Results

The implementation demonstrates the theoretical framework with controlled synthetic data. Results show the advantage of causal inference over traditional approaches:

### Model Performance (30% Budget Constraint)

| Strategy | Avg Uplift (Top 30%) | Improvement vs Random | Key Finding |
|----------|---------------------|----------------------|-------------|
| **Uplift Model** | 0.643 | **+179%** | 81% selections are Persuadables |
| Response Model | 0.435 | +89% | 48% selections are Sure Things (wasted) |
| Random Baseline | 0.230 | 0% | Mixed segments |

**Key Insight**: The uplift model achieves double the improvement of traditional response modeling (179% vs 89%) by correctly identifying and targeting Persuadables while avoiding Sure Things and Sleeping Dogs.

### Generated Outputs

#### Figures (results/figures/)
- `figure1_segment_distribution.png` - Four player segments with targeting recommendations
- `figure2_uplift_calibration.png` - Model calibration (predicted vs actual uplift)
- `figure3_cumulative_gains.png` - Qini curves comparing all strategies
- `figure4_feature_importance.png` - Top predictive features

#### Tables (results/tables/)
- `model_comparison.csv` + `table1_model_comparison.tex` - Performance metrics
- `segment_analysis.csv` + `table2_segment_analysis.tex` - Segment characteristics
- `summary_statistics.csv` - Key numbers for paper

#### Reports
- `results/RESULTS_SUMMARY.txt` - Text summary of all findings

## Configuration

All simulation parameters are controlled via `configs/simulation_config.yaml`:

### Adjust Segment Distribution

```yaml
population:
  total_size: 10000
  segments:
    persuadables: 0.20    # 20% of population
    sure_things: 0.40     # 40% of population
    lost_causes: 0.35     # 35% of population
    sleeping_dogs: 0.05   # 5% of population
```

### Modify Treatment Effects

```yaml
treatment_effects:
  persuadables: 0.70      # +70 percentage points
  sure_things: 0.00       # No effect
  lost_causes: 0.00       # No effect
  sleeping_dogs: -0.40    # -40 percentage points (negative)
```

### Scale Population Size

```yaml
population:
  total_size: 50000       # Increase to 50k players
```

### Tune Model Hyperparameters

```yaml
models:
  uplift:
    n_estimators: 200     # More trees
    max_depth: 15         # Deeper trees
```

After modifying configuration, regenerate results:
```bash
python src/generate_results.py
```

## Methodology

### T-Learner (Two-Model Approach)

The core uplift model implements the T-Learner methodology:

1. **Train separate models**:
   - Treatment model: Trained on players who received bonuses
   - Control model: Trained on players who did not receive bonuses

2. **Predict individual treatment effects**:
   ```
   Uplift(x) = P(Y=1|X=x,T=1) - P(Y=1|X=x,T=0)
   ```

3. **Combine with business value**:
   ```
   Priority = Uplift × LTV × f(Churn)
   ```

### Features (9 core dimensions)

- **Monetary**: Total deposits, average transaction size, spending velocity
- **Behavioral**: Login frequency, session count, days since last login
- **Temporal**: Account age, activity patterns
- **Engagement**: Derived engagement scores

### Evaluation Metrics

- **AUUC**: Area Under Uplift Curve (model quality)
- **Qini Curves**: Cumulative gain visualization
- **Uplift Calibration**: Predicted vs actual treatment effects
- **Improvement over Random**: Percentage gain in targeting efficiency

## Paper Integration

The simulation results are incorporated into Section 9.4 of the updated paper (`ml-bonus-system-paper-with-simulation.docx`). The integration:

- Describes the actual implementation with specific details
- Presents the real simulation results (179% vs 89% improvement)
- Includes new Table 8 with comparative performance metrics
- Provides calibration analysis showing model accuracy
- Emphasizes this demonstrates theoretical mechanisms, not field validation

See `PAPER_UPDATE_SUMMARY.md` for detailed changes.

## Important Limitations

This is a **demonstration with controlled synthetic data**, not empirical validation:

1. **Synthetic Population**: All data is generated with known ground truth segments
2. **Controlled Conditions**: Features designed to correlate with segment membership
3. **Simplified Assumptions**: Stable treatment effects, no strategic player behavior
4. **Pedagogical Purpose**: Demonstrates mechanisms, not real-world proof

The simulation shows what is theoretically possible when assumptions hold. Real-world performance depends on:
- Actual treatment effect heterogeneity in player population
- Quality of available behavioral features
- Model accuracy on production data
- Operational implementation fidelity

**Validation on real gaming data remains future work.**

## Extending the Research

### Add New Features

Edit `src/features.py`:
```python
FEATURE_COLUMNS = [
    'total_deposits',
    'your_new_feature',  # Add here
    # ...
]
```

Update `src/data_generator.py` to generate the feature, then regenerate.

### Test Alternative Uplift Models

Modify `src/uplift_model.py`:
```python
from xgboost import XGBClassifier

# Use XGBoost instead of Random Forest
uplift_model = TLearner(
    base_model=XGBClassifier(n_estimators=200, max_depth=12)
)
```

Or implement S-Learner, X-Learner, or Causal Forests.

### Experiment with Parameters

Modify `configs/simulation_config.yaml` to test:
- Different segment proportions (e.g., 30% Persuadables)
- Varied treatment effect magnitudes (e.g., +0.80 for stronger effects)
- Larger populations (e.g., 100,000 players)
- Alternative feature distributions

### Scale Up

Test with larger populations:
```yaml
population:
  total_size: 100000  # 100k players
```

Note: Runtime and memory requirements scale approximately linearly.

## Reproducibility

All results are deterministic with fixed random seed (42). To reproduce exact results:

```bash
git clone <repository>
pip install -r requirements.txt
python src/generate_results.py
```

The complete pipeline generates identical outputs across runs.

## Documentation

- **QUICK_START.md** - Get started in 5 minutes
- **CLAUDE.md** - Comprehensive developer guide
- **PROJECT_SUMMARY.md** - Architecture and design decisions
- **DELIVERABLES.md** - Complete list of what was built
- **RESULTS_FOR_PAPER.md** - Paper-ready content and statistics

## Use Cases

This codebase serves multiple purposes:

1. **Academic Paper Support**: Provides implementation backing for theoretical framework
2. **Pedagogical Tool**: Teaches uplift modeling concepts with concrete examples
3. **Implementation Template**: Starting point for production systems
4. **Research Foundation**: Base for exploring variations and extensions
5. **Reproducibility**: Allows verification of simulation results

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{stent2026uplift,
  title={A Machine Learning Framework for Optimising Player Bonus Allocation
         Using Causal Inference and Uplift Modelling},
  author={Stent, Mark},
  year={2026},
  note={Technical framework with simulation implementation}
}
```

## License

This code is provided for academic and research purposes.

## Contributing

This is a research implementation accompanying an academic paper. For questions, improvements, or extensions, please open an issue.

## Contact

For questions about this implementation or the underlying framework, please contact the author or open an issue in this repository.

---

## Key Takeaways

1. **Causal inference matters**: Traditional response models achieve only 89% improvement by confusing correlation with causation
2. **Uplift modeling works**: 179% improvement by correctly identifying who responds *because of* treatment
3. **Segments are real**: 81% of selections concentrated in Persuadables (20% of population)
4. **Implementation is feasible**: Complete working system in ~2,500 lines of Python
5. **Framework is extensible**: Modular design supports experimentation and enhancement

**Note**: This is a research simulation demonstrating theoretical mechanisms with controlled synthetic data. All results reflect the behavior of the simulated system under specified assumptions. Validation on production gaming data would be required to quantify real-world performance.

For field deployment, consider:
- Running randomized experiments on actual player population
- Validating feature predictive power with real behavioral data
- Testing robustness to assumption violations
- Implementing responsible gaming safeguards
- Monitoring for model drift and strategic player behavior
