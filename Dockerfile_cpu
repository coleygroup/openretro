FROM continuumio/miniconda3:4.9.2

RUN apt-get update && apt-get -y install gcc g++ make
RUN conda install -c pytorch pytorch=1.8.0 torchserve=0.3.1 cpuonly
RUN conda install -c conda-forge openjdk=11 rdkit
RUN pip install setuptools tqdm OpenNMT-py==1.2.0 dgl==0.4.3 networkx==2.5

ENV CUDA_CODE cpu
ENV TORCH_VER 1.8.0
ENV DGLBACKEND pytorch

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
RUN pip install torch-geometric

WORKDIR /app
COPY . /app/openretro
WORKDIR /app/openretro

# merge split large files
RUN cat ./checkpoints/retroxpert_uspto50k_untyped/retroxpert_uspto50k_untyped.mara* > ./checkpoints/retroxpert_uspto50k_untyped/retroxpert_uspto50k_untyped.mar

# GLN installation
RUN cd ./models/gln_model && pip install -e .

EXPOSE 8080 8081 8082

# Mask out GPUs temperarily, just in case
ENV CUDA_VISIBLE_DEVICES 10
