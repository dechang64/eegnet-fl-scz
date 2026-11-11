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

## 6. Still unverified - check when the first 64-channel file lands

- Whether the 64-channel files' trial ids and ITI match `mergedTrialData.csv`
  row for row (join on `subject`, `trial`, `condition`).
- Whether `TP10` is a usable mastoid reference or already re-referenced away.
- Whether trial counts per subject in the single-trial files equal
  `100 - rejected` from `mergedTrialData.csv`. If they do not, one of the two
  files went through a different rejection pass.
- Whether the documented 1024 Hz is exact (our ERP grid resolves to 1023.96 Hz,
  which is 1024 Hz with a 1024/1023.96 rounding on the time axis).

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
