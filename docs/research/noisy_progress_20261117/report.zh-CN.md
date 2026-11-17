# 2026-11-17 AISHELL-2 noisy progress

Snapshot: 2026-11-17T11:28:31.234549. New real-data pilot began 2026-11-16 16:40:37 CST, formal began17:00:45. Noisy03 completed45/50 (tested frozen validation-best42), noisy04 completed24/50 (tested best23). BF16, fixed512 test, AISHELL-2 speakers disjoint, SIR0..5, target/noise5..20, no RIR. Three legacy-derived cases failed at configuration resolution before any training; fixed and requeued, no result for those cases.

Root cause: inherited configs for noisy01/02/05 lacked data_epoch_offset required by strict HyperPyYAML overrides. Added field and generator fallback, reproduced failure, all5 configs resolve pilot0/formal3, 8 queue/followup tests pass. Source fix957d292. Two running new jobs and two original jobs preserved; old pending backlog temporarily paused to give retries priority.

## noisy03_paper17_local_global_context, tested epoch 42

SI-SNRi dB, only overlap/target-pause; each cell n13..20. No cross-SIR aggregate.

| Noise SNR | SIR0 | SIR1 | SIR2 | SIR3 | SIR4 | SIR5 |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 4.29 | 5.12 | 5.48 | 4.13 | 4.93 | 4.79 |
| 10 | 4.17 | 2.96 | 3.56 | 3.62 | 4.17 | 2.71 |
| 15 | 5.23 | 4.15 | 4.57 | 3.15 | 3.66 | 3.25 |
| 20 | 4.04 | 4.42 | 4.15 | 3.54 | 2.44 | 3.87 |

Target output RMS deviation from clean target, grouped by SIR/SNR: -26.64 to -25.38 dB. Severe attenuation; absence suppression alone does not demonstrate usable target preservation.

| Noise SNR | Absent n | Absent suppression dB | Target+noise only SI-SNRi dB |
|---|---:|---:|---:|
| 5 | 14 | 31.42 | 8.02 |
| 10 | 13 | 29.64 | 5.08 |
| 15 | 14 | 30.27 | 5.50 |
| 20 | 11 | 29.30 | 2.19 |

Training budgets differ; cannot attribute ranking solely to architecture. Fixed test is reporting only, not model selection.

## noisy04_paper16_global_context, tested epoch 23

SI-SNRi dB, only overlap/target-pause; each cell n13..20. No cross-SIR aggregate.

| Noise SNR | SIR0 | SIR1 | SIR2 | SIR3 | SIR4 | SIR5 |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 3.65 | 4.45 | 5.40 | 4.12 | 4.15 | 4.59 |
| 10 | 3.69 | 2.44 | 3.34 | 3.40 | 3.66 | 2.64 |
| 15 | 4.88 | 3.78 | 4.19 | 3.11 | 3.62 | 2.71 |
| 20 | 3.66 | 3.94 | 3.56 | 3.06 | 2.42 | 3.81 |

Target output RMS deviation from clean target, grouped by SIR/SNR: -27.11 to -25.84 dB. Severe attenuation; absence suppression alone does not demonstrate usable target preservation.

| Noise SNR | Absent n | Absent suppression dB | Target+noise only SI-SNRi dB |
|---|---:|---:|---:|
| 5 | 14 | 31.33 | 7.58 |
| 10 | 13 | 29.99 | 3.81 |
| 15 | 14 | 30.76 | 3.93 |
| 20 | 11 | 29.49 | -0.41 |

Training budgets differ; cannot attribute ranking solely to architecture. Fixed test is reporting only, not model selection.
