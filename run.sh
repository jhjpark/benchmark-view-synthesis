echo "BEGIN tiny nerf";
cd tiny-nerf/pytorch;
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-nerf/pytorch/output/tiny_nerf --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-nerf/pytorch/tiny_nerf_pytorch_train.py >> /home/ubuntu/cs246-project/tiny-nerf/pytorch/output/tiny_nerf_ncu.out;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-nerf/pytorch/output/tiny_nerf /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-nerf/pytorch/tiny_nerf_pytorch_train.py >> /home/ubuntu/cs246-project/tiny-nerf/pytorch/output/tiny_nerf_nsys.out;

echo "BEGIN tiny nerf amp";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-nerf/pytorch/output/tiny_nerf_amp --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-nerf/pytorch/tiny_nerf_pytorch_train_amp.py >> /home/ubuntu/cs246-project/tiny-nerf/pytorch/output/tiny_nerf_amp_ncu.out;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-nerf/pytorch/output/tiny_nerf_amp /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-nerf/pytorch/tiny_nerf_pytorch_train_amp.py >> /home/ubuntu/cs246-project/tiny-nerf/pytorch/output/tiny_nerf_amp_nsys.out;
cd ../..;

echo "BEGIN tiny garf gaussian";
cd tiny-garf;
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gaussian --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gaussian_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gaussian_ncu.out;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gaussian /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gaussian_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gaussian_nsys.out;

echo "BEGIN tiny garf gaussian big";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gaussian_big --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gaussian_big_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gaussian_big_ncu.out;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gaussian_big /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gaussian_big_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gaussian_big_nsys.out;

echo "BEGIN tiny garf gelu";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gelu --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gelu_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gelu_ncu.out;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gelu /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gelu_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gelu_nsys.out;

echo "BEGIN tiny garf relu";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_relu --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/relu_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_relu_ncu.out;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_relu /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/relu_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_relu_nsys.out;

echo "BEGIN tiny garf gabor";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gabor --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gabor_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gabor_ncu.out;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gabor /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gabor_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_gabor_nsys.out;

echo "BEGIN tiny garf sin";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_sin --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/sin_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_sin_ncu.out;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_sin /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/sin_train.py >> /home/ubuntu/cs246-project/tiny-garf/output/tiny_garf_sin_nsys.out;
