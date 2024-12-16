FROM mambaorg/micromamba:latest

# create simformer environment
WORKDIR /opt
RUN git clone https://github.com/mackelab/simformer.git
WORKDIR /opt/simformer

# make the conda environment
RUN micromamba env create --file=src/environment.yml
