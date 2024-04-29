# Tiny NeRF

This folder contains the source code used to train and run the Tiny NeRF model from [this](https://arxiv.org/pdf/2003.08934) paper. Their initial implementation was done TensorFlow. Thus, we used [this](https://github.com/krrish94/nerf-pytorch) version that was ported to PyTorch for more optimal performance and to make more direct comparisons with the GARF model, which was also implemented in PyTorch.

The model uses data found in the `data` directory to train. `tiny_nerf_train.py` trains a tiny NeRF model for four epochs using this training data. `tiny_nerf_inference.py` simply runs a randomly initialized model to construct an initial image. Finally, `tiny_nerf_train_amp.py` uses advanced mixed precision (AMP) to train the same model as in `tiny_nerf_train.py`. All of these files can be run by using `python [FILE]`.
