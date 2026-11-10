# D1 playbook — first contact with the real data

Verified end-to-end on a shape-faithful fixture on 2026-11-10 (24 subjects,
long layout, 9 sites, 250 Hz, 3 s centred epochs, condition tokens `1/2/3`):
probe → build → 5-fold training → XAI all ran, and the derived
`event_onset_s = 1.5` proved the latency alignment. Every command below is the
one that was actually exercised. The fixture lives in `_fake_kaggle/`
(never commit it, never cite it).

## 0. Facts fixed by this pass (previously guesses)

| item | was | now |
|---|---|---|
| dataset ref | `byronadams/button-tone-sz` (wrong) | **`broach/button-tone-sz`**, 8.6 GB, CC BY-NC-SA 4.0 |
| cohort | unknown | **81 subjects: 49 SCZ, 32 controls** |
| channels | 5 midline (guessed) | **9 sites: Fz FCz Cz FC3 C3 CP3 FC4 C4 CP4** |
| sampling rate | 500 Hz constant (guessed) | derived from the real time column → `dataset_meta.json` |
| event latency | assumed t=0 | derived (epochs are centred → onset ≈ 1.5 s) |
| labels | CSV by hand | auto-built from `demographic.csv` |

## 1. Download (only the small files first)

```bash
pip install kaggle            # needs ~/.kaggle/kaggle.json (API token)
kaggle datasets download -d broach/button-tone-sz -f ERPdata.csv
kaggle datasets download -d broach/button-tone-sz -f demographic.csv
kaggle datasets download -d broach/button-tone-sz -f mergedTrialData.csv  # optional
unzip "*.zip" -d kaggle_raw/
```

ERPdata.csv + demographic.csv are a few MB. Do **not** download the 8.6 GB
single-trial archive until the ERP-averaged pass has justified it.

## 2. Probe — look before building

```bash
python scripts/probe_dataset.py --erp kaggle_raw/ERPdata.csv \
    --demographics kaggle_raw/demographic.csv --json d1_probe.json
```

Read the JSON. It must show: 81 subjects, the 9 sites present, no unmapped
condition tokens, no subject missing a label, no warnings. If a condition
token or a diagnosis value is unmapped, name it explicitly and rerun:

```bash
python scripts/probe_dataset.py --erp ... --demographics ... \
    --condition-map '{"selfsz":"1","passivesz":"2","buttonsz":"3"}'
# diagnosis values other than SZ/Control etc.:
python scripts/build_dataset.py ... --scz-values "Schizophrenia"
```

## 3. Build

```bash
python scripts/build_dataset.py --erp kaggle_raw/ERPdata.csv \
    --demographics kaggle_raw/demographic.csv --out data
```

Writes `data/<cond>/<cond>_<subject>.parquet`, `data/labels.csv` and
`data/dataset_meta.json` (fs, epoch length, event onset, per-condition counts).
The loader refuses to run without that meta file - that is the point of it.

## 4. First honest run

```bash
python -m src.train --condition selfsz --model eegnet \
    --config configs/default.yaml --labels data/labels.csv
python -m src.train --condition selfsz --model barros_cnn \
    --config configs/default.yaml --labels data/labels.csv
```

Then XAI on held-out subjects only (subject ids from any fold's printout):

```bash
python xai_dissect.py --condition selfsz --labels data/labels.csv \
    --data_dir data --checkpoint results/selfsz_eegnet_fold0.pt \
    --scaler results/selfsz_eegnet_scaler_fold0.npz \
    --holdout-subjects <comma-separated ids of one fold's test subjects>
```

## 5. What to expect, stated before seeing numbers

- `ERPdata.csv` holds **trial-averaged** waveforms: 81 samples per condition,
  1 per subject. Expect wide, embarrassing CIs; a narrow one means a bug.
- With n=81, 5-fold subject-grouped CV puts ~16 subjects per test fold. The
  cluster bootstrap resamples those 16 - an AUC CI of ±0.2 is normal, not a
  failure. Report the CI, not the point estimate.
- sens/spec may disagree wildly at n≈16/fold; the manifest JSON records fold
  composition so any number is traceable.
- This first pass validates protocol and plumbing. The mechanism claim
  (N100 rediscovery) needs the single-trial data (~24 subjects public here,
  more at `broach/buttontonesz2`) and is a separate, later analysis.

## 6. If something raises

That is the designed behaviour. Every raise in `src/data.py` and
`src/ingest.py` names what is unknown and how to fix it. Do not loosen a check
to make data fit; fix the mapping flags or `CHANNELS` with a citation.
