# eegnet-fl-scz

Research scaffold for one question:

> Does an EEGNet trained **only** to classify schizophrenia from the
> button-tone paradigm independently rediscover the N100 corollary-discharge
> signature (attenuated self-generated-tone suppression in SCZ), without ever
> being told what N100 is?

If it does, the model doubles as a mechanism probe, which is a stronger claim
than another point of accuracy. If it does not, that is also a result.

**Status: pipeline verified end-to-end on the real data (81 subjects). First
honest numbers (2026-11-10, trial-averaged ERPs, subject-grouped 5-fold CV):
EEGNet AUC 0.447 +/- 0.074, BarrosCNN 0.516 +/- 0.081 - chance level, as the
dataset author himself predicted for averaged-waveform diagnosis. The
classical N100 suppression contrast (no ML) replicates the phenomenon's
direction in both groups (+1.80 / +1.63 uV) but the group contrast is not
significant with a crude peak metric (p=0.61). Details: `d1_probe.json`,
`scripts/n100_classic_check.py`. Removing the trial averaging does not rescue
the diagnostic claim: with 7836 single trials and subject-grouped CV the
subject-level AUC is 0.595, 95% CI [0.470, 0.720], because ICC(1)=0.04 makes
the effective sample size ~1622 but leaves the number of independent people at
81 (`scripts/single_trial_check.py`). See also [`AUDIT.md`](AUDIT.md) and
[`D1_PLAYBOOK.md`](D1_PLAYBOOK.md).

**Provenance.** Every number above was first computed on a third-party Zenodo
copy and afterwards re-checked against the official Kaggle release. All three
input files are byte-identical (`scripts/check_hashes.py`; digests in
D1_PLAYBOOK section 1a), so these results describe the official data.

## Layout

```
configs/default.yaml       hyper-parameters
src/data.py                button-tone-sz loader (labels from CSV, fails loudly)
src/protocol.py            subject-grouped CV + cluster-bootstrap metrics
src/eegnet.py              EEGNet (Lawhern 2018) with measured feature length
src/barros_cnn.py          1D-CNN baseline, approximation of Barros 2022
src/mechanism_loss.py      condition-difference regularizer (W2, not yet valid)
src/train.py               training entrypoint, writes a run manifest
xai_dissect.py             kernel FFT / spatial weights / virtual ERP
tests/                     guards that encode the audit findings
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# CPU-only torch (much smaller than the default wheel):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pytest tests/ -v
```

## Data

`button-tone-sz` (Kaggle, **`broach/button-tone-sz`**, 81 subjects: 49 SCZ +
32 controls, CC BY-NC-SA 4.0) is **not** in the repo. `ERPdata.csv` provides
trial-averaged ERPs at nine sites (Fz FCz Cz FC3 C3 CP3 FC4 C4 CP4) for all
subjects; single-trial 64-channel data is public for ~24 subjects only.
Build the dataset with `scripts/build_dataset.py` - see
[`D1_PLAYBOOK.md`](D1_PLAYBOOK.md). Expected layout:

```
data/
  selfsz/       selfsz_<subject>.parquet   # or .npy, (channels, time)
  passivesz/    passivesz_<subject>.parquet
  buttonsz/     buttonsz_<subject>.parquet
labels.csv                                   # subject_id,label  (1=SCZ, 0=control)
```

Labels are **never** inferred from filenames: every condition name in this
dataset contains `sz`, so any filename heuristic labels the whole cohort SCZ.
Without `labels.csv` the loader raises.

## Run

```bash
python -m src.train --condition selfsz --model eegnet --labels labels.csv
python -m src.train --condition selfsz --model barros_cnn --labels labels.csv

python xai_dissect.py --condition selfsz --labels labels.csv \
    --checkpoint results/selfsz_eegnet_fold4.pt \
    --scaler results/selfsz_eegnet_scaler_fold4.npz \
    --holdout-subjects s12,s19,s31        # a fold's held-out subjects
```

Outputs land in `results/` (per-fold metrics, scaler, weights, a JSON manifest
with git sha / config / versions) and `xai_results/` (three figures).

## Design commitments

1. **No subject appears in both train and test.** Enforced by construction and by
   an explicit overlap check that raises; a subject carrying two different
   labels is treated as a data fault.
2. **Confidence intervals resample subjects, not epochs.** Ten epochs from one
   subject are not ten independent observations.
3. **Standardisation is fitted on the training fold only** and persisted so XAI
   sees exactly the same transform.
4. **Fail loudly.** Every unresolved constant (`CHANNELS`, `FS`, subject-id
   grammar) raises instead of defaulting to something plausible.

## Not done (D1 checklist)

- [x] confirm file format, channel names/order, sampling rate - resolved
      2026-11-10: 9 sites, fs derived from the real time column into
      `dataset_meta.json`, no more constants
- [x] confirm the subject-id grammar - `<cond>_<subject>` enforced by
      `parse_stem`; the Kaggle subject ids plug in via `build_dataset.py`
- [x] build `labels.csv` from ground-truth metadata - automated from
      `demographic.csv`
- [ ] transcribe Barros 2022's exact architecture into `barros_cnn.py`
- [ ] first real 5-fold run -> first honest AUC  (commands: D1_PLAYBOOK.md)
- [ ] two-site federated extension (FedAvg + DP), week 4

## License

MIT (see `LICENSE`).
