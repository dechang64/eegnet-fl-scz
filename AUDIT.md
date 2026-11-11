# Audit log - eegnet-fl-scz

Audited 2026-11-10 before the first push. Every finding below was reproduced
locally, not inferred from reading. Status legend: FIXED = code changed and a
regression test guards it; DOCUMENTED = deliberately left as-is with a warning.

| ID | Sev | Finding | Status |
|----|-----|---------|--------|
| B1 | high | `pooled_feat_t` closed form ignored block2's temporal conv (+1). **72 values of T in [20,600) disagree with the true feature length** (T=27..30, 59..62, 91..94, ...): either a bogus "too short" ValueError or a head/feature shape mismatch at forward time. Current tests only sampled T=31/63/97/100/320/500, all of which happen to be safe, so the bug was invisible. | FIXED - feature length is now measured by a dry-run forward pass; `test_feat_t_regression_b1` covers all previously broken lengths |
| B2 | high | Bootstrap CI resampled **epochs**, not **subjects**. With ~10 epochs/subject that is pseudo-replication: CIs come out too narrow and any significance claim built on them is unsupported. | FIXED - subject-cluster bootstrap; epoch fallback emits a RuntimeWarning; `test_cluster_bootstrap_is_not_narrower_than_epoch_bootstrap` |
| B3 | high | `_subject_of` returned `stem.split("_")[0]`, so `selfsz_s01.parquet` yielded subject id `"selfsz"`: 81 subjects collapse into 3 pseudo-subjects and the group split becomes a no-op (total leakage). Masked only because the label guard raised first - the first D1 labelling fix would have silently destroyed the protocol. | FIXED - ids parsed/validated, degenerate ids raise; `test_subject_parse_rejects_degenerate_id` |
| B3b | high | Labels were inferred from filenames. In this dataset every condition name contains "sz", so any such heuristic marks the whole cohort SCZ. | FIXED - labels come from an explicit `subject_id,label` CSV; without it the loader refuses to guess |
| B4 | med | `_load_array` reshaped non-2-D arrays with `reshape(arr.shape[-2], -1)` and never checked channel count, length or finiteness; mismatched shapes surfaced as an opaque `np.stack` error much later. | FIXED - 2-D/5-channel/>=32 samples/finite/consistent-shape checks, all fail loudly |
| B5 | med | `confusion_matrix(...).ravel()` unpacked into 4 values crashes when one class is absent; `roc_auc_score` raises likewise. A 5-fold run could die at fold 4 with a data-dependent error. | FIXED - guarded helpers return NaN + warning; `test_single_class_metrics_are_nan_not_crash` |
| B6 | med | No run provenance: results JSON recorded none of git sha / config / library versions / fold composition, so a number could not be traced to the code that made it. Positive-class index was assumed to be 1. A final batch of size 1 would crash BatchNorm. | FIXED - `_manifest()` (git sha, config, versions, fold subject counts, label convention); `pos` resolved from data; singleton batches skipped |
| B7 | med | XAI backward pass hard-coded logit index 1 as "SCZ" (a wrong index inverts every conclusion), analysed training subjects without saying so, and would have plotted wrong scalp positions if the montage changed. | FIXED - `--class-index`, `--holdout-subjects` with explicit warning, montage/channel validation |
| B8 | low | No `max_norm` constraint on the depthwise kernel (Lawhern 2018 uses 1.0); `forward` accepted any tensor shape. | FIXED |
| B9 | med | CI installed only numpy/scikit-learn/pytest, but the test suite imports `src.data` -> `pandas`. CI therefore failed at collection and **never ran a single assertion**; the torch tests were never executed anywhere. | FIXED - dependencies added; separate `guards` and `models` jobs |
| B10 | low | `conftest.py` was a bare comment relying on pytest's rootdir insertion. | FIXED - explicit `sys.path` bootstrap |
| B11 | low | `requirements.txt` listed `tqdm` (unused) and an unpinned `torch`. | tqdm dropped; torch deliberately unpinned with a CPU-wheel note |
| B13 | design | `condition_difference_loss` is label-blind: it pushes self/passive embeddings apart equally for everyone, so it encodes "the conditions differ" but not "they differ less in HC than SCZ". Enabling it as-is would let a paper claim a mechanism constraint it does not implement. | DOCUMENTED (semantics unchanged on purpose - this is a scientific decision, not a bug) |

## Verified state at push time

- `pytest tests/ -v` -> 21 tests, all passing locally (17 guards + 4 model tests
  once torch is present). The earlier claim of "7 passed" circulating with this
  scaffold was never true; the honest count before this audit was 5 passed with
  2 skipped and 1 failing as soon as pandas was installed.
- `python -m py_compile` over every module: clean.
- No credentials, tokens, or data files in the tree (`git ls-files` checked).

## Addendum (2026-11-10, D1 preparation)

- **B14**: `FS` was a hardcoded 500 Hz guess feeding every time axis. Now
  derived from the real time column by the ingester and read from
  `dataset_meta.json` (`resolve_fs`); missing meta raises.
- **B15**: virtual-ERP latency was measured from sample 0 while the dataset's
  epochs are centred on the stimulus; the N100 window would have been shaded
  ~1.5 s off. Onset is now derived from the time column and latency is
  measured from the event.
- **B16**: the documented dataset ref `byronadams/button-tone-sz` does not
  exist; the real one is `broach/button-tone-sz`. The channel montage was also
  wrong (5 midline); the dataset provides nine analysis sites, now in
  `CHANNELS` with an explicit position map.
- Ingest toolchain added (`src/ingest.py`, `scripts/probe_dataset.py`,
  `scripts/build_dataset.py`, `tests/test_ingest.py`): column roles detected
  from aliases, duplicates/missing electrodes/unknown diagnosis values raise,
  fs and event onset derived from data. Exercised end-to-end on a
  shape-faithful fixture (24 subjects, 250 Hz, centred epochs): probe, build,
  5-fold training and XAI all ran; onset derived as 1.5 s. `pytest` -> 46
  passed.

## Addendum 2 (2026-11-10, single-trial analysis)

Question raised: if averaging leaves one sample per subject, what happens
without averaging? `mergedTrialData.csv` (byte-identical to the official
Kaggle release, see D1_PLAYBOOK 1a) carries
23201 usable trials. `scripts/single_trial_check.py` measures what that buys.

  A  single trial, random 5-fold           AUC 0.564   WRONG - leaked
  B  single trial, grouped + aggregated    AUC 0.595   95% CI [0.470, 0.720]
  C  averaged per subject, grouped         AUC 0.527

Protocol A is the pseudo-replication of item B2 in its natural habitat: the
number is not merely optimistic, it is meaningless, because each held-out
trial shares a person with the training set.

Why n does not become 23201: ICC(1) on Fz_N100 is 0.040, so
n_eff = 81 * 97 / (1 + 96 * 0.040) ~ 1622. Trials are not independent
observations. The unit a diagnostic claim generalises over is the person, and
there are 81 of those.

What single trials do buy is measurement reliability: single-trial N100
r=0.040, averaged over 97 trials r=0.822 (Spearman-Brown). Less attenuation,
so a smaller true difference becomes visible - but the ceiling is still set by
81 people. At n=49/32 the smallest AUC reaching 80% power is 0.668, and the
95% CI here tops out at 0.720. The study can only detect a large effect.

Consequence for the paper: a diagnostic-classification claim is not supportable
on this cohort, with or without averaging. A within-subject or
mechanism-oriented claim (does the model recover the suppression difference?)
is the defensible target, and it should be reported with subject-level
clustering throughout.

## What this audit does NOT claim

- No real data has been run. `button-tone-sz` has never been loaded here, so
  `CHANNELS`, `FS`, the subject-id grammar and the file layout remain D1 TODOs.
- `BarrosCNN` is an approximation, not a faithful reimplementation of Barros
  2022 (its docstring already said so; it still says so). Two differences are
  larger than "approximation" and are recorded here so that nobody reads our
  numbers as a failed replication of theirs:
  1. **Input unit.** Barros 2022 fed *single trials* (11,609 trials from 128
     subjects), not subject x condition averages. Their 78% is a single-trial
     model aggregated to the subject by averaging trial probabilities.
  2. **Condition.** They used the *auditory-only* condition (Kaggle
     `condition == 2`) on 5 midline sites (Fz, FCz, Cz, CPz, Pz). This repo's
     D1 command uses `selfsz` (`condition == 1`, self-generated tone) on the 9
     sites that the Kaggle ERP release provides. CPz and Pz are absent from
     that release, which is itself the proof that Barros worked from the raw
     64-channel recordings rather than from `ERPdata.csv`.
  Our AUC ~0.5 therefore does not contradict their 0.78: different input unit,
  different condition, n=81 vs n=128, and 9 sites chosen for the suppression
  contrast rather than for their classifier.
- Accuracy numbers now exist (see `results/`), but they are chance-level and
  describe a different experiment from Barros 2022 - see the two points above.

## Addendum 3 (2026-11-10, the N100 metric was wrong, the effect is there)

Addendum 2 concluded that the N100 suppression contrast did not separate the
groups (HC 1.80 vs SZ 1.63 uV, difference 0.17, p = 0.61). **That conclusion is
withdrawn.** It was an artefact of how this repo measured the component, not a
property of the data.

Reading the dataset owner's own replication notebook
(`broach/replicating-did-i-do-that-paper-analyses-with-r`) exposed three
deviations, and each one alone is enough to bury the effect:

| pipeline | HC | SZ | HC-SZ | p | Cohen d |
|---|---|---|---|---|---|
| canonical: 80-100 ms mean @ Fz/FCz/Cz, self minus button-only | -2.21 | -0.88 | -1.33 | 0.0056 | -0.61 |
| 80-100 ms mean, no button-only subtraction | -2.02 | -1.48 | -0.54 | 0.25 | -0.26 |
| peak in 50-180 ms, no subtraction (what Addendum 2 used) | -1.81 | -1.64 | -0.17 | 0.72 | -0.08 |

The first row reproduces the author's published numbers (2.2 and 0.88 uV) to
two decimal places, which also re-confirms the data copy and the label mapping.

Why the peak metric fails: the N100 is a *mean* over a fixed early window in
this paradigm. Peak-picking a 130 ms-wide window lets trial-to-trial latency
jitter drag the extremum toward whichever of N100 or P200 happens to be larger
for that subject, and averaging those extremes over subjects regresses both
groups toward the same value.

Why the subtraction matters: the button-only condition carries motor potentials
time-locked to the press, and those differ between groups. Skipping the
subtraction leaves that difference inside the "suppression" score, where it
cancels the auditory effect instead of isolating it.

Consequence: **d = -0.61 is equivalent to AUC 0.668**, which is the same
number as the minimum detectable AUC at 80% power in `single_trial_check.json`
(0.668). The effect size and the cohort's detection floor coincide almost
exactly. This is the sharpest available statement about the dataset: at n=81
one can just barely detect this mechanism, and nothing smaller. It is an
argument for a mechanism paper and against a diagnostic classifier - not a
reason to claim no effect.

Carried forward from the same notebook, and not resolved here: subtracting the
button-only epoch can manufacture a positive N1 in subjects whose button-only
condition is itself negative in the window (the author names 60, 61, 64). The
author himself writes "we should revisit the practice of subtracting this
control condition later". Any claim resting on this metric should report both
the corrected and uncorrected values.

Implementation: `scripts/n100_canonical.py` (writes
`n100_canonical_check.json`). `scripts/n100_classic_check.py` keeps its old
metric and its old numbers; it is a provenance check, not a replication.

