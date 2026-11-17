# Original architecture reference implementation plan

- [x] Reproduce missing reference adapter test failure.
- [x] Implement I/O-only adapter, original architecture configuration and snapshot capture.
- [x] Verify exact FP32 output/state equivalence, finite gradients, BF16 real-data train/dev/test.
- [x] Add sixth reference job with pending-slot priority and preserve existing jobs.
- [x] Verify runtime queue, commit code/audit and record deployed hashes.

Verified: original source SHA2568c77071a7f7b34df48f11ae312190d221ef30d8e2d4b720346db8b52d0931850 matches sibling TF_TSE; adapter state/output exactly match FP32 original, finite backward;12 focused server tests pass. BF16 one-step real-data training plus full512 dev and512 test completes. Registered sixth pending case first in queue, effective batch64, pilot3/formal50; supervisors8617/8618 active, current training children retained. Full training has not started yet; next free slot expected in roughly1–1.5h with noisy03 at47/50.
