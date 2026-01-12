# Guide: Integrating Simulation Results into the Paper

## Important Note

The paper integration should be done by **manually editing** the original Word document (`ml-bonus-system-paper-final (10).docx`) to preserve all formatting, tables, references, and structure.

## Why Manual Editing is Required

Automated text conversion (textutil, pandoc, etc.) strips:
- All formatting (fonts, styles, headings)
- Tables and table formatting
- Figure references and captions
- Cross-references and citations
- Page numbers and headers
- Equation formatting
- Professional layout

**DO NOT** convert to text and back - this destroys the document.

## How to Integrate Simulation Results

### Step 1: Open Original Document

Open `ml-bonus-system-paper-final (10).docx` in Microsoft Word or compatible editor.

### Step 2: Update Abstract

**Location**: Abstract section at the beginning

**Find this text**:
> "An illustrative simulation demonstrates the potential magnitude of improvement: uplift-based targeting achieves over 350% higher incremental engagement compared to random targeting"

**Replace with**:
> "Our simulation with controlled synthetic data demonstrates the theoretical mechanisms: uplift-based targeting achieves 179% improvement over random targeting, compared to 89% improvement from traditional response modeling—highlighting the advantages when heterogeneous treatment effects exist"

**Also add** after "worked examples":
> "with controlled synthetic data"

### Step 3: Update Section 9.4 Title

**Location**: Section 9.4

**Current title**: "Illustrative Simulation"

**Change to**: "Implementation Demonstration with Synthetic Data"

### Step 4: Replace Section 9.4.1 Content

**Location**: Section 9.4.1 (Setup)

**Replace the hypothetical setup text with**:

```
We created a synthetic population of 10,000 players with realistic behavioral features
drawn from distributions calibrated to resemble typical gaming populations. The data
generation process implements the four theoretical segments with known treatment effects:

- Persuadables (20%): Baseline engagement 10%, treatment effect +70 percentage points
- Sure Things (40%): Baseline engagement 80%, treatment effect 0
- Lost Causes (35%): Baseline engagement 5%, treatment effect 0
- Sleeping Dogs (5%): Baseline engagement 60%, treatment effect -40 percentage points

Each player has features across multiple dimensions:
- Monetary: Total deposits (varying by segment: $100-2000), average transaction size
- Behavioral: Login frequency (2-25 per month), session count, days since last login
- Temporal: Account age (90-365 days), activity patterns
- Engagement: Derived engagement scores based on activity metrics

Treatment is assigned randomly with 50% probability, and outcomes are generated according
to the segment-specific response functions. The dataset is split 70/30 for training and
testing, stratified by segment to maintain representativeness.
```

### Step 5: Update Section 9.4.2 (Model Implementation)

**Add new subsection** after 9.4.1:

```
9.4.2 Model Implementation

We implement the T-Learner approach using Random Forest classifiers as base learners
(100 trees, max depth 10). Separate models are trained on treatment and control groups,
and uplift is computed as the difference in predicted probabilities. Supporting models
include:

- LTV Model: Random Forest Regressor predicting 12-month customer value
- Churn Model: Random Forest Classifier predicting 30-day churn risk

The composite priority score combines all three outputs: Priority = Uplift × LTV × f(Churn).
```

### Step 6: Update Table 8 (or Create if Missing)

**Location**: Section 9.4.3

**Replace Table 8** with actual simulation results:

| Targeting Strategy | Avg Uplift (Top 30%) | Improvement vs Random | Target Composition |
|-------------------|---------------------|----------------------|-------------------|
| Random (baseline) | 0.230 | 0% | Mixed segments |
| Response model | 0.435 | +89% | 48% Sure Things, 27% Persuadables |
| Uplift model | 0.643 | +179% | 81% Persuadables |
| Full framework | 0.642 | +179% | High-LTV Persuadables prioritized |

**Table caption**:
> "Table 8: Simulation Results by Targeting Strategy. Simulated population n=10,000 players,
> test set n=3,000. Uplift measured as average treatment effect among selected players."

### Step 7: Add Interpretation Section

**Location**: After Table 8, add new section 9.4.4

**Add**:

```
9.4.4 Interpretation and Limitations

These simulation results should be interpreted as demonstration of theoretical mechanisms
under controlled conditions, not as empirical validation of real-world performance.
Several factors make this a pedagogical illustration rather than evidence of field
performance:

First, the true segments and treatment effects are specified by design, ensuring clean
separability. Real player populations may have fuzzier boundaries and noisier treatment
responses.

Second, features were constructed to have reasonable correlation with segment membership.
In practice, available behavioral data may have weaker predictive power.

Third, the synthetic data assumes stable treatment effects and no strategic player
behavior. Real deployments face players who adapt to incentive structures and temporal
drift in effect magnitudes.

Fourth, we use simplistic constant-cost assumptions and binary outcomes. Real optimization
requires modeling varied bonus amounts and continuous revenue outcomes.

Despite these limitations, the simulation demonstrates that the theoretical framework can
be implemented in practice and quantifies the approximate magnitude of advantage when
assumptions hold. The 179% vs 89% improvement gap aligns directionally with field study
findings (Radcliffe and Surry, 2011; Ascarza, 2018).

The complete implementation, including Python code for data generation, model training,
evaluation, and visualization, is available at https://github.com/mark-allwyn/bonus-system
```

### Step 8: Update Key Statistics Throughout

**Find and replace** these vague claims with specific numbers:

- "350%+ improvement" → "179% improvement"
- "over 250-350%" → "179%"
- "substantial improvement" → "179% improvement compared to 89% from response modeling"

### Step 9: Add Implementation Reference

**In the Introduction** (Section 1.3), add:

> "A complete Python implementation demonstrating the framework is available at
> https://github.com/mark-allwyn/bonus-system"

### Step 10: Update Conclusion

**In Section 11**, add before "Future work":

> "A working implementation with controlled synthetic data validates the framework's
> mechanisms and demonstrates 179% improvement over random targeting, compared to 89%
> from traditional response modeling, confirming the theoretical advantages when
> heterogeneous treatment effects exist."

## Key Framing Language

Throughout the paper, ensure simulation is positioned as:

**Use these phrases**:
- "demonstration with controlled synthetic data"
- "illustrates theoretical mechanisms"
- "validates framework implementation"
- "pedagogical illustration"
- "under controlled conditions"

**Avoid these phrases**:
- "proves"
- "empirical evidence"
- "field validation"
- "real-world results"

## After Manual Editing

1. Save the edited document as: `ml-bonus-system-paper-with-simulation.docx`
2. Keep the original: `ml-bonus-system-paper-final (10).docx`
3. Both versions preserved for reference

## Verification Checklist

After editing, verify:
- [ ] All formatting preserved (fonts, styles, headings)
- [ ] All tables intact with proper formatting
- [ ] All figures and figure references correct
- [ ] All citations and references correct
- [ ] Page numbers and headers correct
- [ ] Table of contents updated (if applicable)
- [ ] Cross-references working
- [ ] No broken links or references
- [ ] Professional appearance maintained

## Alternative: Use Word's Track Changes

Consider using Word's "Track Changes" feature to:
- See exactly what was modified
- Allow collaborators to review changes
- Easily revert if needed

## Conclusion

The simulation results are significant (179% vs 89%) and should be integrated into the
paper, but this MUST be done by manually editing the Word document to preserve all the
professional formatting and structure that make it publication-ready.

The automated conversion approach was incorrect - apologies for that error.
