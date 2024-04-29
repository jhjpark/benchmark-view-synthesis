cd tiny-nerf;
mkdir output;
echo "BEGIN tiny nerf";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-nerf/output/tiny_nerf --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-nerf/tiny_nerf_train.py;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-nerf/output/tiny_nerf /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-nerf/tiny_nerf_train.py;

echo "BEGIN tiny nerf amp";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-nerf/output/tiny_nerf_amp --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-nerf/tiny_nerf_train_amp.py;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-nerf/output/tiny_nerf_amp /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-nerf/tiny_nerf_train_amp.py;
cd ..;

cd tiny-garf;
cd gaussian;
mkdir output;
echo "BEGIN tiny garf gaussian";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/gaussian/output/tiny_garf_gaussian --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gaussian/gaussian_train.py;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/gaussian/output/tiny_garf_gaussian /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gaussian/gaussian_train.py;
cd ..;

cd gaussian_big;
mkdir output;
echo "BEGIN tiny garf gaussian big";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/gaussian_big/output/tiny_garf_gaussian_big --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gaussian_big/gaussian_big_train.py;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/gaussian_big/output/tiny_garf_gaussian_big /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gaussian_big/gaussian_big_train.py;
cd ..;

cd gelu;
mkdir output;
echo "BEGIN tiny garf gelu";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/gelu/output/tiny_garf_gelu --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gelu/gelu_train.py;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/gelu/output/tiny_garf_gelu /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gelu/gelu_train.py;
cd ..;

cd relu;
mkdir output;
echo "BEGIN tiny garf relu";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/relu/output/tiny_garf_relu --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/relu/relu_train.py;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/relu/output/tiny_garf_relu /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/relu/relu_train.py;
cd ..;

cd gabor;
mkdir output;
echo "BEGIN tiny garf gabor";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/gabor/output/tiny_garf_gabor --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gabor/gabor_train.py;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/gabor/output/tiny_garf_gabor /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/gabor/gabor_train.py;
cd ..;

cd sin;
mkdir output;
echo "BEGIN tiny garf sin";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project/tiny-garf/sin/output/tiny_garf_sin --force-overwrite --set full --launch-count 100 --kill yes /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/sin/sin_train.py;
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite=true --trace=cuda,nvtx,osrt --python-sampling=true --stats=true --output=/home/ubuntu/cs246-project/tiny-garf/sin/output/tiny_garf_sin /opt/conda/bin/python /home/ubuntu/cs246-project/tiny-garf/sin/sin_train.py;
cd ..;
