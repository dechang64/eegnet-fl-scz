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

## What this audit does NOT claim

- No real data has been run. `button-tone-sz` has never been loaded here, so
  `CHANNELS`, `FS`, the subject-id grammar and the file layout remain D1 TODOs.
- `BarrosCNN` is an approximation, not a faithful reimplementation of Barros
  2022 (its docstring already said so; it still says so).
- Accuracy numbers do not exist yet. Nothing in this repo has produced an AUC.
