# Code Quality

- Separate data loading, transformation, training, evaluation, and inference into testable functions or modules when project size justifies it.
- Keep configuration, paths, random seeds, and model parameters explicit. Avoid environment-specific absolute paths and hidden notebook state.
- Do not execute untrusted project code during static review. Prefer AST inspection and bounded file parsing.
- Handle expected I/O and parsing failures with actionable messages while preserving the original technical error for debugging.
- Pin or constrain important dependencies and document the supported Python version for reproducible environments.
- Never commit API keys, passwords, access tokens, or private data. Read secrets from environment variables or a managed secret store.
- Add focused tests for preprocessing boundaries, schema assumptions, feature generation, and metric calculations.
