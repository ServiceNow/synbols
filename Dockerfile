FROM ubuntu:18.04
#MAINTAINER TODO

# Install Python 3
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    pip3 install --upgrade pip

# Install system dependencies
RUN apt-get install -y fontconfig libcairo2-dev pkg-config

# Install all python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt