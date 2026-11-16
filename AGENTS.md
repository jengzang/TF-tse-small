# Development and experiment provenance

- Keep the local checkout as the primary source tree. Sync code/configuration to the training server; bring any emergency server edits back locally before committing.
- Commit each coherent, verified implementation or experiment change promptly. Do not accumulate multiple days of uncommitted work. Use separate commits for source changes and dated experiment reports when practical.
- Before committing, inspect the diff and run appropriate checks. GPU-dependent verification may run on the training server; record the tested revision/file hashes and actual results. Distinguish unit/smoke verification from completed training experiments.
- Record the deployed commit and file hashes on the server. Running processes may still use earlier loaded source; preserve their checkpoint source snapshots rather than claiming they automatically use newly deployed code.
- Never commit credentials, audio datasets, caches, checkpoints, large generated artifacts, or training logs. Keep dated small audit summaries and reproducibility metadata in Git.
- Preserve existing experiment checkpoints and queue state when deploying or changing scheduling priorities.
