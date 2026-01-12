"""
Master Script to Generate All Publication-Ready Results

Runs the complete pipeline and generates all figures and tables for the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import PlayerDataGenerator
from src.features import FeatureEngineer
from src.uplift_model import TLearner, ResponseModel
from src.ltv_model import LTVModel
from src.churn_model import ChurnModel
from src.scoring import score_players, ScoringPipeline
from src.evaluation import UpliftEvaluator
from src.utils import print_section_header, export_latex_table


def generate_all_results():
    """Run complete pipeline and generate all results."""

    print_section_header("UPLIFT MODELING SIMULATION - COMPLETE PIPELINE", "=")

    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)

    # ========== STEP 1: Generate Synthetic Data ==========
    print_section_header("Step 1: Generating Synthetic Data")

    generator = PlayerDataGenerator()
    df = generator.generate_players()
    df = generator.calculate_true_uplift(df)
    df = generator.generate_ltv_labels(df)
    df = generator.generate_churn_labels(df)

    generator.save_dataset(df, 'data/player_data.csv')

    # Train/test split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['segment']
    )

    generator.save_dataset(train_df, 'data/player_data_train.csv')
    generator.save_dataset(test_df, 'data/player_data_test.csv')

    # ========== STEP 2: Train Models ==========
    print_section_header("Step 2: Training Models")

    # Prepare features
    fe = FeatureEngineer(scale_features=True)
    X_train = fe.prepare_features(train_df, fit_scaler=True)
    y_train = train_df['outcome'].values
    treatment_train = train_df['treatment'].values

    X_test = fe.prepare_features(test_df, fit_scaler=False)

    # Train uplift model (T-Learner)
    print("\nTraining T-Learner...")
    uplift_model = TLearner(random_state=42)
    uplift_model.fit(X_train, y_train, treatment_train)
    uplift_model.save('data/uplift_model.pkl')

    # Train response model (baseline comparison)
    print("\nTraining Response Model...")
    response_model = ResponseModel(random_state=42)
    response_model.fit(X_train, y_train, treatment_train)
    response_model.save('data/response_model.pkl')

    # Train LTV model
    print("\nTraining LTV Model...")
    ltv_model = LTVModel(random_state=42)
    ltv_model.fit(X_train, train_df['ltv'].values)
    ltv_model.save('data/ltv_model.pkl')

    # Train churn model
    print("\nTraining Churn Model...")
    churn_model = ChurnModel(random_state=42)
    churn_model.fit(X_train, train_df['churn_probability'].values)
    churn_model.save('data/churn_model.pkl')

    # ========== STEP 3: Generate Predictions ==========
    print_section_header("Step 3: Generating Predictions")

    test_df['uplift_score'] = uplift_model.predict_uplift(X_test)
    test_df['response_score'] = response_model.predict_response(X_test)
    test_df['ltv_score'] = ltv_model.predict(X_test)
    test_df['churn_pred'] = churn_model.predict_proba(X_test)

    # Generate composite scores
    test_df = score_players(test_df, uplift_model, ltv_model, churn_model, fe)

    # Save predictions
    test_df.to_csv('data/test_predictions.csv', index=False)

    # ========== STEP 4: Generate Evaluation Metrics ==========
    print_section_header("Step 4: Generating Evaluation Metrics")

    evaluator = UpliftEvaluator()

    # Model comparison
    comparison_df = evaluator.compare_models(
        test_df,
        models={
            'Uplift Model': 'uplift_score',
            'Response Model': 'response_score',
            'True Uplift (Oracle)': 'true_uplift'
        }
    )

    print("\n=== Model Comparison Results ===")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('results/tables/model_comparison.csv', index=False)

    # Segment analysis
    segment_analysis = test_df.groupby('segment').agg({
        'player_id': 'count',
        'true_uplift': 'mean',
        'uplift_score': 'mean',
        'outcome': lambda x: x[test_df.loc[x.index, 'treatment'] == 1].mean(),  # Treatment response
        'ltv_score': 'mean',
        'churn_pred': 'mean'
    }).round(3)

    segment_analysis.columns = ['n_players', 'true_uplift', 'pred_uplift', 'treatment_response', 'avg_ltv', 'avg_churn']
    segment_analysis = segment_analysis.reset_index()

    print("\n=== Segment Analysis ===")
    print(segment_analysis.to_string(index=False))
    segment_analysis.to_csv('results/tables/segment_analysis.csv', index=False)

    # Targeting report
    scorer = ScoringPipeline()
    targeting_report = scorer.generate_targeting_report(test_df)

    print("\n=== Targeting Report ===")
    for key, value in targeting_report.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    # ========== STEP 5: Generate Visualizations ==========
    print_section_header("Step 5: Generating Visualizations")

    print("Creating Figure 1: Segment Distribution...")
    evaluator.plot_segment_distribution(
        test_df,
        save_path='results/figures/figure1_segment_distribution.png'
    )

    print("Creating Figure 2: Uplift Calibration...")
    evaluator.plot_uplift_calibration(
        test_df,
        n_bins=10,
        save_path='results/figures/figure2_uplift_calibration.png'
    )

    print("Creating Figure 3: Cumulative Gains (Qini Curves)...")
    evaluator.plot_cumulative_gains(
        test_df,
        models={
            'Uplift Model': 'uplift_score',
            'Response Model': 'response_score',
            'True Uplift (Oracle)': 'true_uplift'
        },
        save_path='results/figures/figure3_cumulative_gains.png'
    )

    # Feature importance visualization
    print("Creating Figure 4: Feature Importance...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    importance_df = uplift_model.get_feature_importance(fe.get_feature_importance_names())
    top_features = importance_df.head(10)

    axes[0].barh(top_features['feature'], top_features['treatment_importance'], color='steelblue')
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Treatment Model Feature Importance', fontweight='bold')
    axes[0].invert_yaxis()

    axes[1].barh(top_features['feature'], top_features['control_importance'], color='coral')
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Control Model Feature Importance', fontweight='bold')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('results/figures/figure4_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========== STEP 6: Generate LaTeX Tables ==========
    print_section_header("Step 6: Generating LaTeX Tables")

    export_latex_table(
        comparison_df,
        'results/tables/table1_model_comparison.tex',
        caption="Performance comparison of targeting strategies on test data (n=3,000 players). Metrics evaluated at 30\\% targeting budget.",
        label="tab:model_comparison"
    )

    export_latex_table(
        segment_analysis,
        'results/tables/table2_segment_analysis.tex',
        caption="Player segment characteristics and treatment effects. True uplift represents ground truth from simulation; predicted uplift shows model estimates.",
        label="tab:segment_analysis"
    )

    # ========== STEP 7: Generate Summary Statistics ==========
    print_section_header("Step 7: Generating Summary Statistics")

    # Calculate key statistics for paper
    uplift_improvement = comparison_df[comparison_df['model'] == 'Uplift Model']['improvement_vs_random'].values[0]
    response_improvement = comparison_df[comparison_df['model'] == 'Response Model']['improvement_vs_random'].values[0]

    summary_stats = {
        'total_players': len(test_df),
        'uplift_vs_random_improvement_pct': uplift_improvement,
        'response_vs_random_improvement_pct': response_improvement,
        'uplift_vs_response_advantage_pct': uplift_improvement - response_improvement,
        'persuadables_pct': (test_df['segment'] == 'persuadables').sum() / len(test_df) * 100,
        'sleeping_dogs_pct': (test_df['segment'] == 'sleeping_dogs').sum() / len(test_df) * 100,
        'avg_true_uplift_persuadables': test_df[test_df['segment'] == 'persuadables']['true_uplift'].mean(),
        'avg_true_uplift_sleeping_dogs': test_df[test_df['segment'] == 'sleeping_dogs']['true_uplift'].mean(),
    }

    summary_df = pd.DataFrame([summary_stats]).T
    summary_df.columns = ['Value']
    summary_df['Value'] = summary_df['Value'].round(2)

    print("\n=== Summary Statistics for Paper ===")
    print(summary_df.to_string())
    summary_df.to_csv('results/tables/summary_statistics.csv')

    # ========== STEP 8: Create Results Summary Document ==========
    print_section_header("Step 8: Creating Results Summary")

    with open('results/RESULTS_SUMMARY.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("UPLIFT MODELING SIMULATION - RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("KEY FINDINGS\n")
        f.write("-" * 70 + "\n")
        f.write(f"1. Uplift model achieved {uplift_improvement:.1f}% improvement over random targeting\n")
        f.write(f"2. Response model achieved {response_improvement:.1f}% improvement over random targeting\n")
        f.write(f"3. Uplift model outperformed response model by {uplift_improvement - response_improvement:.1f} percentage points\n\n")

        f.write("4. PLAYER SEGMENTS IDENTIFIED:\n")
        f.write(f"   - Persuadables: {summary_stats['persuadables_pct']:.1f}% (TARGET - positive uplift)\n")
        f.write(f"   - Sure Things: ~40% (SKIP - no uplift)\n")
        f.write(f"   - Lost Causes: ~35% (SKIP - no uplift)\n")
        f.write(f"   - Sleeping Dogs: {summary_stats['sleeping_dogs_pct']:.1f}% (AVOID - negative uplift)\n\n")

        f.write("PUBLICATION-READY OUTPUTS\n")
        f.write("-" * 70 + "\n")
        f.write("Figures (PNG, 300 DPI):\n")
        f.write("  - figure1_segment_distribution.png\n")
        f.write("  - figure2_uplift_calibration.png\n")
        f.write("  - figure3_cumulative_gains.png\n")
        f.write("  - figure4_feature_importance.png\n\n")

        f.write("Tables (CSV + LaTeX):\n")
        f.write("  - table1_model_comparison.tex\n")
        f.write("  - table2_segment_analysis.tex\n")
        f.write("  - summary_statistics.csv\n\n")

        f.write("=" * 70 + "\n")

    print("\n[OK] All results generated successfully!")
    print("\nOutputs saved to:")
    print("  - Figures: results/figures/")
    print("  - Tables: results/tables/")
    print("  - Summary: results/RESULTS_SUMMARY.txt")

    plt.close('all')


if __name__ == "__main__":
    generate_all_results()
