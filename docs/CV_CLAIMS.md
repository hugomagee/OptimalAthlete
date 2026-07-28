# CV claims this repository supports

Every bullet below is defensible verbatim, with a pointer to the code that proves it.
Anything not on this list is not supported by this repository.

Last verified: 2026-07-28. Reproduce with `python data_loader.py && python models.py && pytest`.

---

## Defensible bullets

### Primary

> **Identified and publicly retracted a leaked R² = 0.84 in my own ML pipeline**, then
> rebuilt it around walk-forward validation and demonstrated the failure mode directly:
> on synthetic data constructed to contain **no** relationship between features and
> outcome, the original pooled random split still reports **R² = 0.906**, while
> walk-forward validation correctly shows the models fail to beat a per-athlete mean
> baseline.

| Clause | Proof |
|---|---|
| The retraction and its cause | [`analysis/recovery_vs_volume.ipynb`](../analysis/recovery_vs_volume.ipynb) §3, [ANALYSIS_NOTES.md](ANALYSIS_NOTES.md) |
| Walk-forward implementation | [`models.py`](../models.py) `walk_forward_predict` |
| No-signal property of the demo data | [`data_loader.py`](../data_loader.py) module docstring — race time is `personal_best + uniform(-0.3, 1.5)` |
| Pooled split reports R² = 0.906 | `models/model_metrics.json` → `naive_pooled_split.random_forest.test_r2` |
| Models lose to the baseline | `models/model_metrics.json` → `walk_forward` (RF MAE 1.063 s vs baseline 0.487 s) |
| Leakage is actually prevented | `tests/test_evaluation_protocols.py::test_walk_forward_uses_only_past_data` |

### Secondary — engineering rigour

> **Built a reproducible, CI-tested ML pipeline** (SQLAlchemy → feature engineering →
> model training → Streamlit dashboard) with 31 tests, ruff linting, and continuous
> integration that also executes the analysis notebook, ensuring published figures cannot
> drift from the code that produced them.

Proof: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), `tests/` (31 tests),
[`pyproject.toml`](../pyproject.toml).

### Secondary — data engineering

> **Found and fixed a silent feature-engineering defect**: rolling "7-day" training
> features were computed over a 7-*row* window, making them independent of training
> frequency — one session count was structurally incapable of varying. Converted to genuine
> calendar windows and added a regression test.

Proof: [`feature_engineering.py`](../feature_engineering.py) `create_training_features`;
`tests/test_evaluation_protocols.py::test_session_counts_reflect_training_frequency_not_row_position`.

### Secondary — statistical practice

> **Established a baseline before claiming model value**: evaluated a "predict the
> athlete's recent average" rule under the identical walk-forward protocol, and reported
> that the ML models do not beat it rather than reporting their absolute R².

Proof: [`models.py`](../models.py) `walk_forward_baseline`, surfaced in the dashboard's
Model evaluation tab.

---

## Claims this repository does NOT support

| Retired claim | Why it is gone |
|---|---|
| **"R² = 0.84 predicting race times"** | Produced by athlete-identity and temporal leakage. The pooled split reproduces a figure of that magnitude on data with no signal at all. Retracted. |
| **"15 engineered features"** | The pipeline builds **12**. Corrected everywhere; the count is recorded in `models/model_metrics.json` and asserted in tests. |
| **"Recovery quality 2.3× more predictive than training volume"** | The provenance audit found those variables are synthetic-generator output. Unverifiable, retracted. |
| **"Training optimisation recommendations for elite athletes"** | The system measures; it does not prescribe. There is no evidence base here for recommending training to anyone. |
| Any predictive-accuracy claim on real athlete data | The bundled data is synthetic and contains no signal by construction. The methodology is ready for real wearable data; that data is not yet in hand. |

---

## If asked about the retraction in an interview

> My first version reported R² = 0.84 predicting race times from training data. When I
> audited it I found the split pooled athletes randomly, so the model was mostly learning
> which athlete was racing — about 95% of the variance in pooled race times is between
> athletes rather than within them. I retracted the number and rebuilt the evaluation
> around walk-forward validation. Then I made the failure reproducible: the bundled demo
> data has no relationship between features and race time by construction, and the old
> protocol still reports R² = 0.906 on it, while walk-forward correctly shows the model
> losing to a per-athlete mean baseline.

The point is not that the first attempt was wrong — it is that the error was found by
systematic self-auditing, and that the response was to publish the retraction and build a
demonstration of the mechanism rather than quietly restate the number.
