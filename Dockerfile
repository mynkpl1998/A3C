FROM ubuntu:18.04

MAINTAINER "Mayank Kumar Pal (mayank15147@iiitd.ac.in)"

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install apt-utils wget nano

ARG DEBIAN_FRONTEND=noninteractive

# Minconda python3.6 Installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /root/miniconda3/bin:$PATH
RUN conda update conda
RUN conda install python=3.6.8

# Install PyTorch Deep learning Framework
RUN conda install pytorch-cpu torchvision-cpu -c pytorch

WORKDIR /home/

# Install full open ai gym dependencies and library
RUN apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb git-all
RUN git clone https://github.com/openai/gym.git
RUN cd gym
RUN pip install -e . '.[atari]'

# Copy Application Code to the Container
COPY . /home/

# Set the working directory to the code directory
WORKDIR /home/

# Install Dependencies
RUN pip install -r requirements.txt

EXPOSE 6006