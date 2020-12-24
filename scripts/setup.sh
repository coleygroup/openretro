TORCH_VER=1.7.0
CUDA_VER=10.1
CUDA_CODE=cu101

conda create -y -n openretro python=3.6 tqdm
conda activate openretro

conda install -y pytorch=${TORCH_VER} torchvision torchaudio cudatoolkit=${CUDA_VER} -c pytorch
conda install -y rdkit -c rdkit

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
pip install torch-geometric

# GLN installation, make sure to install on a machine with cuda
cd ./models/gln_model
pip install -e .
cd ../..