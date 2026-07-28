# Analysis notes — recovery vs volume

Defense document for [`analysis/recovery_vs_volume.ipynb`](../analysis/recovery_vs_volume.ipynb).
Every methodological choice, the alternatives considered, and the hard questions an
interviewer should ask — with pointers to where the notebook answers them.

## The short version

The analysis set out to verify two CV claims — "R²=0.84" and "recovery quality was 2.3×
more predictive of performance than training volume" — and instead found that (a) the
recovery/volume variables in the database are synthetic-generator output, (b) the R²
figure was an artifact of athlete-identity and temporal leakage, and (c) under honest
validation neither predictor block shows out-of-sample signal. Both claims were retracted.
The notebook's lasting value is the audited, CI-tested methodology, ready for real data.

## Methodological choices and why

### Why a provenance audit before any modeling (notebook §1b)

The database had a history: a synthetic generator (`data_loader.py`) wrote to the same
path, and `add_real_data.py` explicitly mixed real races with generated training data. Any
analysis that skipped provenance would risk publishing conclusions about random numbers.
The battery tests for generator fingerprints (full-precision floats, bounded integer HRV,
zero day-to-day autocorrelation, exact generator window, presence of known real results).
*Alternative considered:* trusting the data and adding a caveat. Rejected — the audit takes
30 lines and its result (predictors are synthetic) invalidates the headline question, which
is exactly the kind of thing one must find out first.

### Why walk-forward validation, not k-fold CV (notebook §4)

The prediction task is a forecast: predict the *next* race from training history. Random
k-fold lets future races into the training set of past test races, which is precisely the
leakage found in the original pipeline (§3). Walk-forward (expanding window, fit strictly
on earlier races) mirrors deployment and is the standard evaluation for small time series.
*Alternatives considered:* blocked/purged CV (defensible, but with n≈7 the blocks would be
1–2 races anyway — walk-forward is the same idea with a cleaner story); leave-one-out CV
(rejected: leaks time in both directions).

### Why permutation importance, not coefficients or impurity importance (notebook §4)

- Impurity-based (tree) importances are biased toward high-cardinality features and are
  computed on *training* data — they measure what the model used, not what predicts.
- Coefficients require a linear model and standardization choices, and say nothing about
  out-of-sample value.
- Permutation importance on held-out predictions is model-agnostic, expressed in seconds of
  OOS MAE, and supports **grouped** permutation: the whole recovery block is permuted
  jointly, preserving its internal correlations — the fair way to compare a 4-variable
  block against a 1-variable block.

*Alternative considered:* SHAP. Rejected as primary — SHAP explains model output, not
held-out predictive value, and would lend false sophistication to a 7-row dataset.

### Why Ridge as the primary model (notebook §4)

With at most 6 training races per fold, any flexible learner memorizes noise. Standardized
Ridge (α=10) is the most honest primary: linear, heavily regularized, deterministic. The
Random Forest appears only as a robustness variant. *Alternative considered:* Gaussian
process with a time kernel — attractive but unfitable at this n without strong priors doing
all the work.

### Why the difference is primary and the ratio secondary (notebook §5)

The CV quotes a ratio ("2.3×"). Ratios of noisy quantities explode when the denominator
nears zero — and volume's importance was positive in only ~9% of bootstrap resamples, so
the ratio is undefined most of the time. The difference (recovery − volume, in seconds of
OOS MAE) is stable, interpretable, and was therefore reported with its CI as the primary
comparison; the ratio's instability is itself reported as a finding.

### Why a pairs bootstrap over the OOS evaluation set (notebook §5)

The uncertainty that matters for the claim is "how much could this comparison move under
resampling of the evaluation races?". The bootstrap resamples OOS races with fold models
held fixed, which is documented in the notebook — refit-level uncertainty would be larger,
so the reported CIs are a *lower bound* on honest uncertainty. *Alternative considered:*
full refit bootstrap (breaks the time ordering that walk-forward depends on); jackknife
(n=4 OOS points makes it degenerate).

### Why a multiverse table (notebook §6)

"Recovery" and "volume" have no single canonical definition. Testing one definition invites
cherry-picking; testing ~12 combinations (model class × definitions × windows × influence
checks × a pooled athlete-centered variant) and summarizing in aggregate ("0% of variants
show meaningful signal") makes the negative result robust rather than an artifact of one
specification.

### Why effect sizes in seconds (notebook §7)

Importance rankings don't answer "what is 1 SD of better recovery worth?". A descriptive
standardized Ridge with a pairs bootstrap gives that in race-time units, with CIs that
(honestly) span zero.

## The 10 hard interview questions

1. **"Your predictors turned out to be synthetic. Why publish the analysis at all?"**
   Because the audit *is* the result: I claimed a number, went to verify it, and found the
   data couldn't support any version of it. Notebook §1b–§1c documents the evidence; §10
   states what would make the question answerable. Killing the claim publicly is the
   portfolio point.

2. **"Why should I believe the provenance battery over your memory of collecting data?"**
   The fingerprints are mechanical: HRV is an integer series in exactly the generator's
   ±10% band with lag-1 autocorrelation ≈ 0 (real HRV is strongly autocorrelated), and the
   session history spans exactly one default generator window. §1b prints each check with
   its evidence.

3. **"Walk-forward with a first training set of 3 races — isn't that meaningless?"**
   Nearly, and the notebook says so (§9, limitation 5). The alternative — random CV — is
   *worse than meaningless* because it manufactures skill via leakage (§3 quantifies this).
   Small honest numbers beat large leaky ones.

4. **"How do you know the original R²=0.84 was identity leakage and not real skill?"**
   §3: 95.6% of pooled outcome variance is between athletes, and a baseline that predicts
   each athlete's training-set mean — knowing nothing about form — achieves R²≈0.91,
   *beating* the Random Forest (≈0.64). The model was mostly a noisy athlete classifier.

5. **"Why grouped permutation rather than permuting features one at a time?"**
   The recovery block's variables are correlated; permuting them individually lets the
   model lean on the others, understating block importance and making a 4-vs-1 comparison
   incoherent. Joint permutation destroys the block's information while preserving its
   internal structure (§4).

6. **"Your bootstrap holds the fold models fixed. Isn't that cheating?"**
   It conditions on them, and the notebook discloses the direction of the bias (§5): true
   uncertainty is larger than shown. Given the conclusion is already "no signal, wide CIs",
   extra variance strengthens rather than weakens it.

7. **"With ~12 multiverse variants, couldn't you have found a positive result by chance?"**
   Yes — which is why no single cell is quoted. The reported summaries are aggregates
   (share of variants favoring recovery: 10%; share with any meaningful signal: 0%), and
   §9 lists multiple comparisons explicitly. No p-values are reported anywhere.

8. **"The races themselves — are they real?"**
   Partially unverifiable (§9, limitation 6): 12 of 15 have the two-decimal signature of
   hand-entered results, 3 have generator fingerprints and are dropped in a sensitivity
   variant. The known-real 2025 results documented in the repo are absent from the DB.

9. **"What would change your conclusion?"**
   Real logged wearable data over 12+ months, pre-registered exposure definitions, and the
   same pipeline (§10). The notebook runs end-to-end on a synthetic stand-in in CI, so the
   methodology is ready the day real data exists.

10. **"You found the model doesn't beat 'predict my recent average'. Why ship it?"**
    Because the deliverable is the measurement system, not the current score. §4 reports
    the naive baseline next to the model precisely so future improvements are judged
    against the right bar.

## What the CV can now say

See notebook §10 for the quotable sentence, and [CV_CLAIMS.md](CV_CLAIMS.md) for the exact
bullets this repository supports verbatim, each with a pointer to the code that proves it.

Anything citing "R²=0.84" or "2.3×" should be removed; the defensible replacement is the
audited negative result plus the methodology. Since this document was first written, the
pipeline itself has been rebuilt around walk-forward validation, so the contradiction it
described — an audit that demolished the pooled split sitting beside a `models.py` that
used one — no longer exists. See the README changelog.
