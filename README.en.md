# speech_tftse

[中文 README](README.md)

This repository contains a TensorFlow-TSE/TFGridNet-style research implementation for **target speaker extraction (TSE)**. Given a speech mixture and an enrollment condition for the target speaker, the model reconstructs the target waveform. The source tree keeps several related lines of work together:

- the original two-channel TF-TSE baseline;
- a genuine single-channel T0 variant;
- offline bundle datasets, AISHELL-2 noisy cases, and structural ablations;
- training, resume, evaluation, listening, and experiment provenance records.

This is a **source-and-experiment repository**, not a self-contained release with speech corpora, pretrained weights, caches, checkpoints, or a configured server environment. Keep datasets, checkpoints, training logs, and large listening exports outside Git. The repository stores configuration templates, compact reports, tests, and reproducibility metadata.

## Current status

| Item | Current contract |
| --- | --- |
| Sample rate | 16 kHz; the speaker encoder also requires 16 kHz |
| Target speakers | One target speaker; no PIT |
| Enrollment | A 256-D ResNet34_yk embedding, normally shaped `[1, 256]` |
| Two-channel baseline | Mixture `[B, 2, N]`; reference and enrollment audio are mono |
| Single-channel T0 | Mixture `[B, 1, N]`, embedding `[B, 1, 256]`, output `[B, N]` |
| Output | Mono target waveform with the target reference length |
| Main spectral path | 512-point STFT; training also uses 512/256/128 multi-resolution terms |
| Hardware | CPU supports smoke/tests; full training and quality evaluation normally need CUDA/ROCm |
| Streaming | Not a complete streaming implementation: centered STFT and whole-segment normalization are used |

Parameter count depends on the configuration. The original small TFGridNet backend has about 298k parameters; the unified mono model in `models/model_tfgridnet_mono.py` has about 298,138 parameters. The speaker encoder is not included in the TSE parameter count. Always report the configuration, checkpoint contract, and experiment result together.

## Quick start

### 1. Clone and create an environment

```bash
git clone https://github.com/jengzang/TF-tse-small.git
cd TF-tse-small
python3 -m venv .venv
. .venv/bin/activate
```

Install a `torch`/`torchaudio` build matching the local CPU, CUDA, or ROCm platform first, then install the repository dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the additional legacy online room-mixing dependencies only when that data path is needed:

```bash
python -m pip install -r requirements-train.txt
```

`requirements-train.txt` includes the base requirements and pins `pyroomacoustics==0.7.3`. The legacy `WhamRoom` code uses the pre-0.8 `Source.get_rir` API, so Python 3.10/Linux and a working C++ build toolchain are recommended. Offline bundle training, inference, and most tests do not require this package.

### 2. Run checks without the real speech corpora

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/inspect_cases.py --case all --output /tmp/tftse-case-profile.json
```

The test suite checks data contracts, model shapes, losses, inference, case registration, and the training entry point. `inspect_cases.py` constructs the cases and writes parameter/profile information on CPU. Neither command proves CUDA throughput, long-run convergence, or speech quality.

### 3. Run offline inference

With an existing 256-D embedding:

```bash
python infer.py \
  --config config/config_tfgridnet_all.yaml \
  --checkpoint /path/to/checkpoint.pth.tar \
  --mix /path/to/two_channel_16k.wav \
  --embedding /path/to/embedding.npy \
  --output /tmp/target.wav
```

Or extract the enrollment condition from a mono 16 kHz enrollment recording:

```bash
python infer.py \
  --config config/config_tfgridnet_all.yaml \
  --checkpoint /path/to/checkpoint.pth.tar \
  --mix /path/to/two_channel_16k.wav \
  --enroll /path/to/enrollment_16k_mono.wav \
  --speaker-checkpoint /path/to/model_19.pkl \
  --output /tmp/target.wav
```

Use exactly one of `--embedding` and `--enroll`. Inference validates 16 kHz audio, channel count, finite/nonzero embeddings, checkpoint contracts, and output length. Inputs must be non-empty finite audio readable by `soundfile`.

## Repository layout

```text
.
├── config/                 # Main, mono, case, and server data configurations
├── dataset/                # Legacy corpus, offline bundle, and AISHELL-2 noisy data
├── models/                 # TSE models, mono variants, case backends, vendored FLA
├── trainer/                # Training loop, precision context, checkpoint saving
├── utils/                  # STFT, enrollment, audio cache, losses, room helpers
├── emb_extract/            # Optional ResNet34_yk speaker encoder implementation
├── scripts/                # Data preparation, queues, evaluation, migration, reports
├── tests/                  # Unit, contract, and smoke tests
├── docs/research/          # Dated reports, summaries, and provenance JSON
├── docs/superpowers/       # Design documents and execution plans
├── train_dp_multi.py       # Main training entry point
├── infer.py                # Offline inference entry point
├── requirements.txt        # Base dependencies
└── requirements-train.txt  # Extra legacy online-mixing dependencies
```

Recommended reading order:

1. `config/config_tfgridnet_all.yaml`: complete two-channel baseline configuration;
2. `models/model_TFGridNet_k1_stage2_multi_v5.py`: `Tar_Model.forward`, STFT layout, and GridNet blocks;
3. `models/local/dcunet.py`: frontend convolutions, normalization, and conditional channels;
4. `dataset/bundle_data.py` and `dataset/corpus_mix.py`: offline and online sample contracts;
5. `train_dp_multi.py` and `trainer/trainer_dp_v1.py`: loading, initialization, resume, saving, and source snapshots;
6. `utils/loss/losses.py`: SI-SNR, valid-length, and multi-resolution spectral losses;
7. `infer.py` and `scripts/evaluate_case.py`: inference and full-length evaluation;
8. `docs/research/`: results and data/checkpoint provenance.

## Model and data contracts

### Two-channel baseline

The default configuration instantiates `models.model_TFGridNet_k1_stage2_multi_v5.Tar_Model`. The mixture is `[B, 2, N]`, the target reference is `[B, N]`, and the enrollment condition is `[B, 1, 256]` or an equivalent batched form. The model emits one target waveform. `mono_policy: duplicate` can duplicate a mono bundle mixture for the old two-channel interface; it does not turn the model into a genuine mono architecture.

The ordinary training branch combines negative SI-SNR, real/imaginary/magnitude L1 terms at 512/256/128 FFT resolutions, and a small waveform-energy term. Validation primarily uses negative SI-SNR. Because the target is specified by enrollment, permutation-invariant training is not used.

### Single-channel T0

`config/config_mono_legacy.yaml`, `config/config_mono_fixed.yaml`, and `config/config_mono_unified.yaml` instantiate `models.model_tfgridnet_mono.Tar_Model`:

- mixture input is `[B, 1, N]` and is never duplicated;
- enrollment is `[B, 1, 256]`;
- output is `[B, N]`;
- `mono_policy: mono` rejects a two-channel mixture;
- the unified configuration enables `padding_aware_loss` and `target_absent_loss`, computes loss over valid samples, and records output RMS separately for absent-target samples;
- legacy/fixed preserve source loss behavior for layout/behavior comparisons and do not claim a completed reproduction of the old online-mixing training recipe.

The model pads to an STFT hop and crops back to the original length. Short inputs are supported, but centered STFT, whole-segment normalization, and padding effects mean this is not a complete streaming model. A two-channel checkpoint must be explicitly migrated before use with a mono model:

```bash
python scripts/migrate_dual_to_mono.py \
  --checkpoint /path/to/dual.pth.tar \
  --output /tmp/mono_legacy_init.pt
```

The migration sums the two audio kernels in the first layer, preserves the conditional kernel, checks all other keys/shapes, and writes a migration report without optimizer state. Numerical equivalence only applies when the two old input channels are exactly identical; it does not establish lossless real-world two-channel-to-mono conversion.

### Checkpoint contract

Training outputs normally live under `chkpt/<config.name>/`, with logs under `logs/<config.name>/`. The training entry point saves the resolved configuration, model source snapshot, and SHA-256 manifest. A running process has already loaded its source and does not change when the repository is synchronized later; reports must record the revision, configuration, and checkpoint hash actually used.

- `continue_from`: resume the full training state, including model, optimizer, epoch/step state, and other recoverable state;
- `init_from`: load model weights only and start a new optimizer/experiment counter;
- the two options are mutually exclusive;
- an existing experiment directory is protected unless a valid `continue_from` is supplied;
- mono checkpoints carry input/layout contracts, and inference/resume rejects missing or incompatible contracts.

## Data preparation

The repository supports two main data paths. Both require 16 kHz audio. Replace paths in a local or server configuration rather than using absolute paths from the author's environment.

### Legacy corpus / online mixing

The default configuration uses the legacy data path when `dataset_type` is absent. It expects speaker-to-utterance maps, precomputed speaker embeddings, noise SCPs, room/RIR parameters, and fixed development/test SCPs. Template locations include:

- `data/AISHELL1/`, `data/AISHELL2/`, `data/Librispeech/`, `data/aidatatang/`, and `data/3D-Speaker/`;
- `data/wham_noise/`, `data/room/`, and `data/rir_noise/`;
- `train_spk2utts_path`, `train_embd_path`, `dev_*_scp`, and `test_*_scp`.

Use `scripts/prepare_corpora.py` to build a speaker-disjoint index, an enrollment embedding cache, and fixed dev/test bundles from LibriSpeech, AISHELL-1/2, and related directories. Its defaults point to a particular server and must be overridden:

```bash
python scripts/prepare_corpora.py \
  --output-dir /path/to/corpus_bundle \
  --libri-root /path/to/LibriSpeech \
  --aishell1-root /path/to/aishell1/wav \
  --aishell2-scp /path/to/aishell2_scp \
  --aishell2-root /path/to/aishell2/wav \
  --speaker-checkpoint /path/to/model_19.pkl \
  --device cuda:0
```

The script checks 16 kHz mono audio, split-disjoint speakers, enrollment recordings of at least 2.9 seconds, and CPU/GPU embedding agreement. It cannot replace human checks of speaker identity, data rights, or physical reference alignment.

### Offline bundle

A bundle uses three ID-aligned SCP files:

```text
sample_id /absolute/path/to/mix.wav
sample_id /absolute/path/to/ref.wav
sample_id /absolute/path/to/enrollment.wav
```

The ID sets of `mix`, `ref`, and `aux` must be identical. A two-channel bundle has mix shape `[2, N]`; a mono bundle has `[1, N]` and uses `mono_policy: mono`. Reference and enrollment audio are mono. Audio must be non-empty, finite, and 16 kHz; mix and aux must not be silent.

Embedding caches are commonly saved with `torch.save` and have at least:

```python
{"embeddings": {"sample_id": tensor_of_shape_[1,256]}}
```

It is recommended to store speaker-checkpoint hash, index hash, sample rate, and `ref_duration` metadata as well. Existing embeddings avoid loading the speaker encoder for every sample; `enrollment_mode: waveform` extracts the condition from the aux WAV at read time.

Bundle path template:

```bash
cp config/cases/data.example.yaml config/cases/data.server.yaml
# Edit bundle_* paths and ref_duration in data.server.yaml.
```

`bundle_root` can rewrite the fixed marker prefix used by a data export when the server mount point changes. Other relative paths are resolved relative to the data configuration file.

## Training, resume, and experiment cases

### Baseline training

After verifying every data path in `config/config_tfgridnet_all.yaml`:

```bash
python train_dp_multi.py --config config/config_tfgridnet_all.yaml
```

The default is 400 epochs, 16 kHz, 10-second segments, and batch size 8. Outputs go to `chkpt/baseline/dcunet_tfgridnet_v5_run01/`. If that experiment directory already exists and `continue_from` is empty, training refuses to overwrite it; change `name` or provide the intended resume checkpoint.

### Unified mono training

Prepare a real mono bundle with aligned mix/ref audio and a `[1,256]` embedding cache, then edit the `bundle_*` paths in `config/config_mono_unified.yaml`:

```bash
python train_dp_multi.py --config config/config_mono_unified.yaml
```

Use legacy/fixed configs for behavior comparisons and unified for valid-length and absent-target experiments. CPU smoke tests and parameter counts do not establish real speech quality, speaker-selection quality, or chip-level real-time performance.

### Server cases

`config/cases/case00_*.yaml` through `case09_*.yaml` cover independent small/wide TFGridNet, mix-skip, FiLM, temporal FLA, and temporal BiLSTM comparisons. Configure shared data paths with:

```bash
cp config/cases/data.example.yaml config/cases/data.server.yaml
# Edit data.server.yaml.
python scripts/run_case.py \
  --case case00 \
  --data-config config/cases/data.server.yaml \
  --device cuda \
  --precision bf16 \
  --batch-size 2 \
  --max-steps 2000
```

For formal cases, add options such as `--epochs`, `--resume`, `--init-from`, `--evaluate-test`, and `--samples-per-epoch` as appropriate. See the complete interfaces with:

```bash
python scripts/run_case.py --help
python scripts/run_server_queue.py --help
```

`--init-from` is a model-only warm start; the optimizer starts fresh. `--resume` and configuration-level `continue_from` restore training. Keep each case separately named, with its config/source snapshot, and do not overwrite existing checkpoints or change the meaning of queued work.

### Evaluate a checkpoint

Full-length evaluation of a mono bundle:

```bash
python scripts/evaluate_case.py \
  --config config/config_mono_unified.yaml \
  --checkpoint /path/to/temp_best.pth.tar \
  --mix-scp /path/to/test/mix.scp \
  --ref-scp /path/to/test/ref.scp \
  --aux-scp /path/to/test/aux.scp \
  --embedding-cache /path/to/test/embeddings.pt \
  --output /tmp/tftse-eval.json \
  --device cuda \
  --precision fp32
```

The report includes input/output SI-SNR, SI-SNR improvement, gain, waveform MAE, absent-target output RMS, and SHA-256 values for the config, manifests, and checkpoint. SIR/SNR groups require metadata JSON. Selecting a checkpoint by test metrics or tuning the test sample list invalidates an independent test claim.

## Losses, evaluation, and interpreting results

- Ordinary branch: negative SI-SNR plus real/imaginary/magnitude L1 at multiple FFT resolutions and a small waveform-energy term;
- `padding_aware_loss`: negative SI-SNR and masked spectral terms only over valid samples;
- `target_absent_loss`: absent-target samples stay out of the active SI-SNR mean and are reported through output RMS/relative mixture energy;
- evaluation should use a fixed dev/test manifest and validation-best checkpoint; test results must not be used to select the checkpoint and then reported as independent;
- report SIR, noise SNR, active/absent target counts, gain, and pause intervals separately instead of relying on one aggregate mean.

Existing reports are under [`docs/research/`](docs/research/). Useful entry points include:

- [server case handbook](docs/research/2026-11-10-server-cases.zh-CN.md);
- [T0 mono implementation and verification](docs/research/2026-11-10-t0-implementation.zh-CN.md);
- [TF-TSE audit design](docs/research/2026-11-10-tf-tse-audit-design.zh-CN.md);
- [AISHELL-2 noisy progress](docs/research/noisy_progress_20261117/report.zh-CN.md);
- [requested SIR/SNR comparison](docs/research/requested_sir_20261118/report.zh-CN.md);
- [original-model reference design](docs/superpowers/specs/2026-11-17-original-reference-design.md).

Listening WAVs, HTML pages, and temporary JSON are local/server products. Historical compact reports are kept in the repository, but that does not mean every generated artifact should be committed.

## Verification and troubleshooting

### Recommended checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/inspect_cases.py --case all --output /tmp/tftse-case-profile.json
git diff --check
git status --short
```

Before GPU training, run a short job with a small sample budget to verify configuration parsing, exact SCP ID alignment, complete embedding caches, checkpoint contracts, and mixed-precision stability. Passing a CPU smoke check only establishes that the code path and input contracts are viable; it does not establish real-data quality.

### Common issues

**Data is missing or an absolute path is invalid**: copy the relevant `*.example.yaml`, replace every `data/...` and server path, and avoid editing paths inside historical reports.

**SCP IDs do not match**: compare the sets of IDs in `mix`, `ref`, and `aux`; do not pair files by sorted position. `dataset/bundle_data.py` rejects duplicate IDs, empty SCPs, and set mismatches.

**Invalid embedding shape**: inference accepts `.npy` shapes `(256,)`, `(1,256)`, or `(1,1,256)`. A bundle cache item must be `(1,256)`, finite, and nonzero.

**Two-channel/mono mismatch**: `input_channels`, `mono_policy`, model class, and checkpoint contract must agree. Never load a two-channel checkpoint directly into a mono model; migrate it first.

**Experiment directory already exists**: the training entry point protects prior results. Change `name` for a new experiment or fill in the correct `continue_from`; do not delete an old checkpoint to bypass the guard.

**pyroomacoustics fails to build**: it is an extra dependency for legacy online room mixing. Check Python 3.10, a compiler, and SciPy compatibility. Inference and offline bundle paths can omit it.

**Results do not match current source**: the process loaded its source and configuration at startup. Inspect the checkpoint snapshot/hash, recorded Git revision, and server deployment record. A later source synchronization does not mutate an already running process.

## Server synchronization and Git policy

The local checkout is the primary source tree. Deploy code/configuration to the training server explicitly, and bring emergency server edits back locally before committing. For each deployment or experiment, record:

- the Git commit and SHA-256 hashes of important files;
- the actual configuration, model source, dependency versions, and device;
- the starting checkpoint revision and whether the run was a warm start or a full resume;
- start/end times, queue state, and report paths.

Do not commit credentials, datasets, caches, checkpoints, large generated audio, training logs, or virtual environments. `artifacts/` is ignored for local listening/progress products; historical tracked compact reports remain as they are. Commit each coherent verified implementation or experiment promptly, inspect the diff, and run appropriate checks before committing.

## Related documentation

- [Development and experiment provenance rules](AGENTS.md)
- [Chinese agent prompt](AGENT_PROMPT.zh-CN.md)
- [Designs and execution plans](docs/superpowers/)
- [Research reports](docs/research/)
- [GitHub repository](https://github.com/jengzang/TF-tse-small)

## License and third-party code

`models/vendor/fla/` contains the vendored FLA source together with its `LICENSE` and `SOURCE.json`. Follow the original licenses and access policies for third-party datasets, ResNet34_yk speaker weights, and historical TF-TSE code. This README does not grant redistribution rights to datasets or model weights.
