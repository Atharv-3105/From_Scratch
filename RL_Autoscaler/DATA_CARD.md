# DATA_CARD.md: Azure Functions Trace 2019 (Sampled)

## Provenance
- **Source:** Azure Public Dataset (Functions 2019) - Sampled Subset
- **Paper:** Shahrad et al., "Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider", USENIX ATC 2020.
- **License:** CC-BY 4.0 (Must cite the paper in README and public posts)
- **Original Download URL:** (Locked - HTTP 409 Public access not permitted)
- **Actual Download URL:** https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset
- **Date Downloaded:** [FILL IN TODAY'S DATE]
- **Archive Size:** 460.72 MB (Uncompressed ~1.5 GB)

## Rationale for Subset
The full Azure Functions 2019 dataset exceeds 200GB. This Kaggle-hosted subset is a random sample of applications where the creator verified that the workload distributions (rare vs. hot functions, invocation concentration) remain a good match to the original ATC 2020 paper. This aligns with the project's constraint to use a stratified subset for local, CPU-bound development.

## Actual Contents (Consolidated .feather format)
Instead of 40 separate daily CSVs, this subset provides 3 consolidated columnar files:

1. **invocations_per_function_md.feather**: 618,545 rows, 1445 cols
   - Function invocation counts per minute (likely 1440 columns for 24 hours)
   
2. **function