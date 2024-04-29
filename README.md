# Benchmarking View Synthesis Methods

In this repository, you will find all the code used for this project, as well as the final paper. This includes the Python JPEG compression and decompression found in `jpeg`, the NeRF training and inference code found in `tiny-nerf`, and the GARF training and inference code found in `tiny-garf`. Each individual folder has a specific README that indicates the source of the source code, as well as more in-depth descriptions.

We used `run.sh` to collect results from NVIDIA Nsight for each model. The script runs NVIDIA Nsight Compute for 100 kernel launches. Then, it runs NVIDIA Nsight Systems on the entire model end-to-end. We used a version of each model that trains for four epcohs. Although this is relatively small, each epoch looks quite similar, so we just needed enough to see how the epochs overlapped.
