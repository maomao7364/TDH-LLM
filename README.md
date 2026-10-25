# TDH-LLM: Data and Resources

This repository provides the data and preprocessing scripts used in the study **“TDH-LLM: A Pre-Trained Large Language Model with Temporal Decomposition and Horizon-Aware Decoding for Multi-step Traffic Flow Forecasting.”**

TDH-LLM combines temporal decomposition, graph-based node representations, a pre-trained GPT-2 backbone, and horizon-aware decoding for multi-step traffic flow forecasting.

## Datasets

- **PeMSD7-derived:** A 50-station traffic dataset adopted from prior research, distinct from the standard PeMS07 benchmark.
- **PeMS08:** A publicly available traffic dataset containing observations from 170 sensors.

Both datasets are sampled at 5-minute intervals.

## Data Availability

The source data and preprocessing scripts for both datasets are available in this repository. The scripts define the data preparation procedures and train/validation/test splits used in the experiments.

The original datasets remain subject to their respective usage and redistribution terms. Please acknowledge the original data sources when using these resources.
