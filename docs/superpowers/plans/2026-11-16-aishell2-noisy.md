# AISHELL-2 noisy automatic follow-up plan

Goal: prepare a verified, deferred five-direction noisy speech training stage, preserving current experiments.
Architecture: optional corpus filtering/SIR arguments, separate noise dataset and fixed-manifest preparation, validation-only selector and durable waiting scheduler.

- [x] Add failing dataset tests: AISHELL2-only sources, no reverse SIR, RMS calibration, disjoint noise, reproducibility, invalid audio rejection.
- [x] Implement optional base corpus arguments and isolated noisy subclass; preserve defaults.
- [x] Find and audit real environmental noise; generate fixed 512 dev/512 test, separated speakers and noise files.
- [x] Add train/run_case optional route, snapshot source files and follow-up runner.
- [x] Add tests for dependency barrier and five distinct validation-only choices.
- [x] Run regressions and BF16 real-data smoke; sync code and prepared manifests.
- [x] Launch locked supervisor in waiting state; verify no existing tasks interrupted and next-stage cannot start early.

Verification 2026-11-16: 30 regression tests passed on server; BF16 one-step training, full 512-example dev validation and 512-example test inference passed with protocol metadata assertions. Follow-up PID13308 holds exclusive lock and reports waiting_for_all_existing_tasks. Existing paper supervisor PID26802 continues 5 completed/3 running/14 pending. Selection remains unfrozen until dependency barrier clears. Fixed validation BundleDataset enrollment_mode propagation before raw cases begin; regression reproduced 256 vs46400 failure and now passes. No existing raw formal results were available/invalidated.
