# Reproducibility

- Set and record random seeds for Python, NumPy, model libraries, data splitting, sampling, and hyperparameter search when those components are stochastic.
- A random seed improves repeatability but does not guarantee bit-for-bit results across hardware, library versions, parallel execution, or nondeterministic accelerators.
- Declare the supported Python version and constrain runtime dependencies in `requirements.txt`, `pyproject.toml`, or an environment lock file.
- Record dataset identity, source, retrieval date, schema version, and transformation version so a result can be traced to the exact input.
- Keep model parameters, preprocessing choices, split strategy, thresholds, and metric configuration in explicit configuration or logged experiment metadata.
- Avoid absolute machine-specific paths; use project-relative paths, configuration, environment variables, or command arguments.
- Preserve a rerunnable entry point that starts from raw or versioned data rather than relying on an interactive session's hidden state.
- Separate secrets from reproducibility metadata: document required variable names but never commit API keys or credentials.
- Capture warnings, package versions, and important platform limitations when exact reproducibility is not possible.
