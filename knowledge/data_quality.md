# Data Quality

- Profile row/column counts, missingness, duplicates, inferred types, invalid ranges, category cardinality, and target distribution before modeling.
- Treat missing values according to their mechanism and business meaning; compare missing and non-missing groups before deletion or imputation.
- Large-scale `dropna` can silently change the population and class distribution. Measure removed rows and key segment shifts before accepting it.
- Fit imputers and learned cleaning rules on training data only, then reuse the fitted objects for validation, test, and inference data.
- Check duplicates at the intended business grain. Exact row duplicates and repeated entity-event records require different decisions.
- Validate schema, units, date/time zones, encodings, category vocabularies, and target availability at ingestion boundaries.
- Investigate outliers with domain ranges, robust summaries, and segment views before removing them; rare valid cases may be the cases the model must handle.
- A type such as `object` can hide numeric strings, mixed date formats, sentinel values, or inconsistent categories; explicit parsing should report failures.
- For classification, report class counts and minority ratios. Accuracy alone can be misleading on imbalanced data; consider stratification and class-sensitive metrics.
- Very small datasets make validation unstable, while very wide datasets increase overfitting and leakage risk; state sample-to-feature limitations explicitly.
- Report data limitations explicitly; clean-looking aggregates do not prove completeness, representativeness, or correct labeling.
