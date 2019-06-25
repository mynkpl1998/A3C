FROM ubuntu:18.04

MAINTAINER "Mayank"

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install apt-utils wget nano

# Copy Application Code to the Container
COPY . /home/

# Set the working directory to the code directory
WORKDIR /home/

# Minconda python3.6 Installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /root/miniconda3/bin:$PATH
RUN conda update conda
RUN conda install python=3.6.8

# Install Dependencies
RUN pip install -r requirements.txt

# Install PyTorch Deep learning Framework
RUN conda install pytorch-cpu torchvision-cpu -c pytorch
