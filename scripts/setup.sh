TORCH_VER=1.7.1
CUDA_VER=10.1

conda create -y -n openretro python=3.6 tqdm
conda activate openretro

conda install -y pytorch=${TORCH_VER} torchvision torchaudio cudatoolkit=${CUDA_VER} -c pytorch
conda install -y rdkit -c rdkit

cd ./models/gln_model
pip install -e .
cd ../../..