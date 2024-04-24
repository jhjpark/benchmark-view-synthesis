echo "BEGIN tiny nerf";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project-main/tiny-nerf/pytorch/output/tiny_nerf --force-overwrite --set full /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-nerf/pytorch/tiny_nerf_pytorch_train.py >> /home/ubuntu/cs246-project-main/tiny-nerf/pytorch/output/tiny_nerf_ncu.out;
nsys profile --stats=true --output=/home/ubuntu/cs246-project-main/tiny-nerf/pytorch/output/tiny_nerf /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-nerf/pytorch/tiny_nerf_pytorch_train.py >> /home/ubuntu/cs246-project-main/tiny-nerf/pytorch/output/tiny_nerf_nsys.out;

echo "BEGIN tiny garf gaussian";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gaussian --force-overwrite --set full /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/gaussian_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gaussian_ncu.out;
nsys profile --stats=true --output=/home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gaussian /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/gaussian_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gaussian_ncu.out;

echo "BEGIN tiny garf gaussian big";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gaussian_big --force-overwrite --set full /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/gaussian_big_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gaussian_big_ncu.out;
nsys profile --stats=true --output=/home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gaussian_big /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/gaussian_big_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gaussian_big_ncu.out;

echo "BEGIN tiny garf gelu";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gelu --force-overwrite --set full /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/gelu_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gelu_ncu.out;
nsys profile --stats=true --output=/home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gelu /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/gelu_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gelu_nsys.out;

echo "BEGIN tiny garf relu";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_relu --force-overwrite --set full /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/relu_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_relu_ncu.out;
nsys profile --stats=true --output=/home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_relu /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/relu_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_relu_nsys.out;

echo "BEGIN tiny garf gabor";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gabor --force-overwrite --set full /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/gabor_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gabor_ncu.out;
nsys profile --stats=true --output=/home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gabor /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/gabor_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_gabor_nsys.out;

echo "BEGIN tiny garf sin";
sudo /usr/local/cuda-12.1/nsight-compute-2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_sin --force-overwrite --set full /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/sin_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_sin_ncu.out;
nsys profile --stats=true --output=/home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_sin /opt/conda/bin/python /home/ubuntu/cs246-project-main/tiny-garf/sin_train.py >> /home/ubuntu/cs246-project-main/tiny-garf/output/tiny_garf_sin_nsys.out;
