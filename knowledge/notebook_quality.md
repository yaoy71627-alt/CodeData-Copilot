# Python and Notebook Quality

- A notebook should run successfully from a clean kernel in top-to-bottom order; execution counts that jump backward suggest hidden ordering dependencies.
- Avoid relying on variables, imports, files, or model objects created by cells that are not part of the documented execution path.
- Move reusable loading, transformation, training, and evaluation logic into functions or modules once the notebook becomes difficult to rerun or test.
- Keep parameters near the top or in explicit configuration rather than scattering mutable global state across cells.
- Replace hard-coded absolute paths with project-relative paths or configurable inputs that work on Linux and Windows.
- Use explicit, bounded data reads and writes; declare encodings, expected schemas, index handling, and overwrite behavior.
- Measure the effect of `dropna`, filtering, joins, and deduplication. Each operation should report how many records and which segments were removed.
- Preserve preprocessing order: split first, fit learned transformations on training data, then transform validation and test data.
- Do not mix exploratory display side effects with the only copy of production logic; keep a deterministic path that can be executed without manual cell intervention.
- Clear saved outputs that contain private data, tokens, local paths, or excessively large artifacts before sharing a notebook.
