# The two Kaggle datasets, studied before downloading 39 GiB

Written 2026-11-10, from the Kaggle API (not from the dataset pages) plus the
sources of the notebooks attached to the datasets. Everything here is checkable
with the commands at the bottom.

## 1. Identity

| | `broach/button-tone-sz` | `broach/buttontonesz2` |
|---|---|---|
| title | EEG data from basic sensory task in Schizophrenia | EEG data from sensory task in Schizophrenia (2) |
| subtitle | Button press and auditory tone ERPs from 81 human subjects | Additional ERP data from human subjects |
| owner | Brian Roach (UCSF / SFVAMC, co-author of Ford 2013/2014) | same |
| licence | CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 |
| current version | v6, 2019-01-31 | v4, 2019-01-31 |
| files / unpacked | 45 / 19.38 GiB | 41 / 19.54 GiB |
| zip download | 8.03 GiB | 8.11 GiB |
| downloads / votes / kernels | 12,093 / 249 / 24 | 3,255 / 36 / 14 |
| subject CSVs | 1-24 and 66-81 = **40** | 25-65 = **41** |

The two datasets **partition the 81 subjects**; neither is complete on its own.
The description of the first one still says "twenty four subjects" - true when
it was written at v3 (2017-11-17), never updated after the 2019 revisions. Any
pipeline that reads only the first dataset silently analyses half the cohort.

## 2. What one subject file actually contains

Read off `columnLabels.csv` (285 B) and `time.csv` (47 KB), then confirmed in
Roach's own MNE loader:

- **No header row.** 74 columns = `subject, trial, condition, sample` + 70
  channels. The 70 = 64 scalp EEG + `VEOa VEOb HEOL HEOR` (EOG) + `Nose` +
  `TP10` (right mastoid). `columnLabels.csv` supplies the names.
- **3072 samples per trial**, 1024 Hz, -1500 to +1499 ms. Sample index 1537
  (1-based) is time 0.
- **`condition`**: 1 = button+tone, 2 = playback, 3 = button alone. Taken from
  the `event_id` dict in the official loader, not guessed.
- **No `rejected` column** - outlier trials were already dropped before upload.
  `mergedTrialData.csv` is the only place a rejection flag survives (3.3% of
  its 23,986 rows).
- **Not raw.** Nine preprocessing steps were applied (re-reference to averaged
  earlobes, 0.1 Hz high-pass, channel interpolation, epoching, -100..0 ms
  baseline, CCA for muscle/line noise, trial rejection, ICA component removal,
  within-trial channel interpolation). There is no route back to continuous
  data, and no ICA weights are published.

`ERPdata.csv` (120,228,250 B) is the average across trials per subject x
condition, 9 sites, same 3072-point grid: 81 x 3 x 3072 = 746,496 rows.
`mergedTrialData.csv` (15,361,541 B) is 23,986 trials x 36 scalars
(9 sites x N100 / P200 / B0 / B1). **It is not a waveform table** - the only
trial-level waveforms are the 64-channel files. B0 sits around 0.06 and B1
around 8.2 in magnitude, consistent with the intercept and slope of a linear
trend fitted per channel per trial, but the author never defines them; treat
them as undocumented.

## 3. What the authors themselves say about using it

Four notebooks are by `broach`. They are the closest thing to a spec, and two of
them undercut claims made on top of this data:

**`replicating-did-i-do-that-paper-analyses-with-r`** (2018, 64 votes) is the
canonical analysis of the 81-subject cohort. It fixes the metric:

- N100 = mean amplitude in **80-100 ms**, averaged over Fz, FCz, Cz.
- The button+tone condition has the **button-only condition subtracted**
  (`cond1 - cond3`) to remove motor potentials time-locked to the press.
- suppression = playback - corrected button+tone.
- GLS with compound symmetry, subject as random factor.
- Result: suppression **HC 2.2 uV (95% CI 1.8-2.6), SZ 0.88 uV (0.6-1.2)**,
  condition x group p < .005, CIs non-overlapping.
- Two honest caveats in the notebook: the baseline is not exactly zero over
  -100..0 ms (needs the inclusive window), and the cond3 subtraction can
  manufacture a *positive* N1 (subjects 60, 61, 64) - "we should revisit the
  practice of subtracting this control condition later".

**`predicting-single-trial-n100-starter`** (2020) is the author trying to predict
single-trial N100 from `mergedTrialData.csv` with a GLS model. His verdict:
"our single trial N100 predictions are pretty poor", and "the RMSE of this
approach is so poor that it should be an easy benchmark to beat". It also finds
that the suppression effect **grows across the run** (under 1 uV at trial 1,
nearly 3 uV by trial 100) and that ITI matters - i.e. trials are not
exchangeable, which rules out shuffling them freely.

**`getting-started-with-mne-python-and-subs-1-to-24`** (2019, updated 2024, 107
votes) is the reference loader: `header=None`, reshape to
`(n_trials, 3072, 74)`, drop the first 4 columns. Its decoding example is
*within-subject*: CSP + logistic regression separating button+tone from
playback trials, 5-fold CV inside one person. That is the use the data was
published for - a mechanism probe, not a diagnostic classifier. Note the
notebook no longer runs: `read_montage`, `_prepare_topo_plot` and the
`plot_topomap(..., epos)` signature were all removed in MNE >= 0.23.

## 4. What the community did with it

31 kernels attached to dataset 1, 19 to dataset 2, many of them forks. Eleven
distinct ones were pulled and read:

| pattern | kernels | note |
|---|---|---|
| trial-level `train_test_split(..., shuffle=True)` | 7 | includes the highest-voted (361 votes) and its two forks |
| trial-level `StratifiedKFold` | 1 | grouped by nothing |
| trial-level `cross_val_score` | 1 | |
| within-subject condition decoding | 1 | the official loader |
| N100 statistics (no ML) | 1 | the R replication |
| **subject-grouped CV** | **0** | |

The dominant recipe pools all trials from all 81 subjects into one array and
splits 80/20 at random. Trials from the same person then sit on both sides, so
the model can identify the subject instead of the diagnosis. The most recent
kernel (2026-10-01, "SZ Button Tone - 40") does exactly this - 420 hand-made
per-trial features, all trials concatenated, `random_split` - and normalises
each trial by its own maximum before pooling. Accuracies from this family are
not comparable to Barros 2022's 78%, which used a subject-wise 10-fold split
and averaged trial probabilities per subject.

Positive detail: several of these kernels loop `1..81` and fall back from the
first dataset to the second, which independently corroborates the partition in
section 1.

## 5. Consequences for this repo

1. **The N100 group difference is real and we had been erasing it.** Following
   the canonical recipe on the data already on disk reproduces Roach's numbers:
   HC -2.21 uV vs SZ -0.88 uV, Welch p = 0.0056, d = -0.61. Without the cond3
   subtraction the difference collapses to -0.54 uV (p = 0.25), and with
   peak-picking it vanishes (-0.17 uV, p = 0.72). `scripts/n100_canonical.py`
   implements the canonical version; `n100_classic_check.py` stays as the
   deliberately crude provenance check it always was.
2. **d = -0.61 converts to AUC 0.668**, which is the same number as the
   minimum detectable AUC at 80% power computed in `single_trial_check.json`
   (0.668). The effect and the detection floor coincide: this cohort can just
   barely see the mechanism, and cannot see anything smaller. That is a
   publishable statement about the dataset, and it is the argument for a
   mechanism paper rather than a classifier paper.
3. **Single trials do not rescue the classifier.** An AUC-equivalent of 0.668
   is the ceiling for a perfect, noise-free measurement of this one feature;
   the between-subject ICC of 0.040 means trial count buys reliability, not
   sample size.
4. **Do not download 39 GiB to start.** Pull per subject with `%2F` in the URL
   (203 MiB compressed each) and build the loader against 4-6 subjects first.
   Disk on this machine: 44 GiB free on C:, 32 GiB on D:.

## 6. Verified against real 64-channel files (2026-11-11, subjects 1 2 3 67 68 69)

Downloaded 6 subjects (HC 1/2/3, SCZ 67/68/69, ~205 MiB zip each) to
`D:/eegnet-data/single_trial` and ran `scripts/verify_single_trial.py`.
Every check passed for all six subjects (`verify_single_trial.json`):

- Files are **headerless**: 74 numeric columns exactly as `columnLabels.csv`
  (4 index + 64 scalp + VEOa/VEOb/HEOL/HEOR/Nose/TP10). The earlier MNE
  notebook's `pd.read_csv(..., header=None)` on a file with a header does not
  apply to the shipped files.
- Trial blocks: each trial is a contiguous 3072-row block, ~289 trials per
  subject, in **presentation order, not sorted by trial id** (subject 1
  starts with trial 100). Join by `(trial, condition)`, never by position.
- Trial sets match `mergedTrialData.csv` `rejected == 0` rows exactly
  (subject 1: 289 = 289; condition split 100/94/95 identical). So the raw
  files contain only accepted trials — `100 - rejected` confirmed.
- fs is exactly **1024 Hz** (`time.csv`: 0.9766 ms spacing, event at sample
  1537 = -1500 ms relative to file start; 3072 samples span 3.0 s).
- Per-trial N100 recomputed from raw waveforms (mean 80-100 ms at Fz/FCz/Cz)
  matches `mergedTrialData.csv`'s `*_N100` columns to **0.0000 uV** for all
  trials of all six subjects — the feature table is derived from these very
  waveforms with the same convention.
- Still open: whether `TP10` is a usable mastoid reference (column exists;
  re-referencing state not yet checked) — matters only if we montage-plot.

## 7. How to re-check any of this

```bash
export KAGGLE_KEY=...   # a token, then: Authorization: Bearer $KAGGLE_KEY
# metadata, versions, licence, kernel count
curl -s -H "Authorization: Bearer $KAGGLE_KEY" \
  https://www.kaggle.com/api/v1/datasets/view/broach/button-tone-sz
# full file list with per-file byte counts (paginated)
curl -s -H "Authorization: Bearer $KAGGLE_KEY" \
  "https://www.kaggle.com/api/v1/datasets/list/broach/button-tone-sz?pageSize=100"
# the notebooks quoted above
curl -s -H "Authorization: Bearer $KAGGLE_KEY" \
  "https://www.kaggle.com/api/v1/kernels/pull?userName=broach&kernelSlug=predicting-single-trial-n100-starter"
```

Note that `datasets/view` and `datasets/list` accept basic auth
(`-u user:key`), while `kernels/*` only accepts the bearer header and answers
401 otherwise.

The per-file endpoint needs the subject to appear twice, because on Kaggle each
subject is a directory holding one file:

```bash
# correct (203-227 MiB each, 81 subjects, 17 GiB total):
.../api/v1/datasets/download/broach/button-tone-sz/10.csv%2F10.csv
# 404 even though the subject exists:
.../api/v1/datasets/download/broach/button-tone-sz/10.csv
```

`scripts/download_single_trial.py` drives this, validates every archive's CRC,
and skips subjects already on disk.

## 8. All 81 subjects: the effect reproduces, the classifier does not

Data: all 81 subjects (32 HC labelled 0, 49 SZ labelled 1), 23,201 trials,
`D:/eegnet-data` 17 GiB, downloaded 2026-11-11. Three scripts, three questions.

### 8.1 Which condition code is which

The raw files label conditions 1/2/3 with no legend. Correlating each code's
per-subject mean 80-100 ms amplitude against the already-split
`data/{selfsz,passivesz,buttonsz}` parquet epochs, over 81 subjects:

|        | selfsz | passivesz | buttonsz |
|--------|--------|-----------|----------|
| code 1 | 1.000  | 0.704     | 0.301    |
| code 2 | 0.704  | 1.000     | 0.044    |
| code 3 | 0.301  | 0.044     | 1.000    |

**code 1 = button+tone (self), code 2 = passive playback, code 3 = button only.**
Codes 1 and 2 correlate at 0.70 because both carry the tone-evoked response;
code 3 does not. This is now an empirical fact rather than an assumption.

### 8.2 The canonical N100 survives the trip through the raw files

`scripts/single_trial_features.py` recomputes the published metric straight from
the raw trials (80-100 ms mean @ Fz/FCz/Cz, passive minus self-minus-button):

| source | HC | SZ | HC-SZ | p | d | AUC |
|---|---|---|---|---|---|---|
| parquet epochs (`n100_canonical_check.json`) | -2.21 uV | -0.88 uV | -1.33 | 0.0056 | -0.61 | 0.668 |
| raw single trials, all 81 subjects | -2.21 uV | -0.88 uV | -1.33 | 0.0056 | -0.61 | 0.668 (rank 0.691) |

Identical to two decimals. The raw release is the same data, and the effect the
paper reports is reproducible from it without touching the parquet copies.

### 8.3 One feature vs an end-to-end network

| decoder | input | protocol | subject-level AUC |
|---|---|---|---|
| logistic regression | 1 feature: 80-100 ms mean, 7,836 button+tone trials | subject-grouped 10-fold | 0.533 (trial) / 0.617 (subject mean) |
| EEGNet (unchanged) | 5 midline channels x 500 ms, 23,201 trials | subject-grouped 10-fold | **0.376**, acc 0.457 |

Two facts, both needed:

1. **The signal is real but thin.** A single targeted feature reaches 0.617 at
   subject level under grouped CV - close to the 0.668 ceiling implied by
   d = -0.61, and it is the *same* feature the paper reports.
2. **The end-to-end network does not find it, and lands below chance.** AUC 0.376
   on 81 subjects is roughly two standard errors under 0.5; the per-fold
   accuracies (0.33, 0.62, 0.62, 0.25, 0.38, 0.50, 0.50, 0.50, 0.50, 0.38) show
   no consistent direction. Under grouped CV the model is mostly predicting the
   0.60 prior (mean P(SZ) = 0.611 for HC, 0.535 for SZ), i.e. it collapsed onto
   the base rate rather than onto the inverted effect.

Consequence for the write-up: the honest claim is about the **cohort and the
measurement**, not about a classifier. A 5-channel, 500 ms, 81-subject problem
with d = -0.61 is below the detection floor of an end-to-end network trained on
23k trials; a hand-specified feature sits right at that floor. Anyone reporting
90%+ on this dataset is splitting trials, not subjects (section 4).

Caveat carried forward: one seed for the 10-fold protocol. The below-chance
point rests on the gap to the 0.617 feature baseline, not on the precise 0.376.
The 81-fold LOSO run (`--protocol loso`) is the remaining check.

Artifacts: `single_trial_full_10fold.json`, `single_trial_features.json`,
`D:/eegnet-data/single_trial_features.npz` (per-trial amplitudes, 124 KiB, so
later analyses skip the 17 GiB re-read).
