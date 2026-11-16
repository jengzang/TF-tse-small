# Paper ablations implementation plan

Goal: Implement and launch the approved 22-case backlog, keep four GPUs supplied, retain reproducible controlled comparisons.
Architecture: Separate model/enrollment/loss modules and a new queue; existing variants default to unchanged behavior. No repository Git metadata, edit in place and preserve previous checkpoints/config/source snapshots.

- [ ] Write red tests for LCA branches, CLN, hierarchical temporal block, raw-context dependency and zero-enrollment rejection, identity/critic gradients, GRL signs, strict budget.
- [ ] Write red tests for consistent raw-enrollment loading/corpus/bundle support and queue slot/PID/OOM recovery rules.
- [ ] Implement independent model_paper_ablation and enrollment/loss helpers; opt-in trainer integration with default zero extra weights.
- [ ] Add raw input routing to corpus/bundle/evaluation/inference without changing embedding defaults. Separate raw tests verify enrollment and mixture sources remain independent.
- [ ] Generate 22 explicit configs and immutable experiment registry, pilot/formal data offset.
- [ ] Implement durable resource-aware queue, capture actual source/config/data manifests and BF16 smoke each config.
- [ ] Server red/green tests, regression on all old variants and audio guards, all-parameter BF16 finite gradients and parameter counts; no trained-result claims from smoke.
- [ ] Sync hashes, launch queue, observe real pilot iteration and confirm no unintended old process interruption. Deliver matrix/status paths and precise running/pending counts.
