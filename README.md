# Hackathon Repository Summary (HaulMark Dumper Fuel Consumption)

This repository contains multiple iterations of a hackathon solution for predicting dumper fuel consumption from high-frequency telemetry, with optional spatial context and a mix of “clean” (no leakage) and “ultra-optimized” (high OOF but leakage-prone) approaches.

## What this repo is for

- Predict fuel consumption for unseen time windows from telemetry.
- Engineer features that aggregate noisy, irregular telemetry into stable daily/shift summaries.
- Train models with vehicle-aware validation (GroupKFold by vehicle) and generate Kaggle-style submissions.
- Produce diagnostics (OOF plots, feature importances, route benchmarks).
