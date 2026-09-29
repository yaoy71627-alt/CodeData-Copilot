# Data Leakage Prevention

- Perform train/test split before any operation that learns statistics, categories, thresholds, selected columns, or sampling rules from data.
- Fit scalers, imputers, encoders, feature selectors, dimensionality reduction, and resampling only on the training partition; validation and test partitions only call `transform`.
- Put learned preprocessing inside a scikit-learn `Pipeline` so every cross-validation fold fits preprocessing on that fold's training subset.
- Feature selection must run inside the training fold. Selecting features on the full dataset leaks validation/test signal even if the model is fitted later.
- Prevent target leakage by removing direct labels, post-outcome fields, future events, target-derived aggregates, and proxies that would not exist at prediction time.
- Define a feature-availability cutoff: every feature value must be observable at or before the prediction timestamp.
- For repeated people, devices, stores, or patients, use group-aware splitting so the same entity does not appear on both sides of evaluation.
- For time-dependent problems, use chronological or rolling splits; random train/test split can leak future patterns into earlier predictions.
- Keep the final test set untouched until preprocessing, model family, thresholds, and hyperparameters are fixed.
- A suspicious `fit_transform` call is static evidence of risk, not proof of leakage; verify its position relative to splitting and the object passed to it.
