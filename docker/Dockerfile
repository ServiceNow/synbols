FROM ubuntu:18.04 AS base
#MAINTAINER TODO

ARG GOOGLE_FONTS_COMMIT=ed61614fb47affd2a4ef286e0b313c5c47226c69

# Install Python 3
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    pip3 install --upgrade pip

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl fonts-cantarell fontconfig git icu-devtools ipython3 jq libcairo2-dev libhdf5-dev pkg-config ttf-ubuntu-font-family unzip wget

# Install all python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download CIFAR10
RUN git clone https://github.com/YoongiKim/CIFAR-10-images.git /images

# Install all Google fonts and extract their metadata
RUN wget -O google_fonts.zip https://github.com/google/fonts/archive/${GOOGLE_FONTS_COMMIT}.zip && \
    unzip google_fonts.zip && \
    mkdir -p /usr/share/fonts/truetype/google-fonts && \
    find fonts-${GOOGLE_FONTS_COMMIT} -type f -name "*.ttf" | xargs -I{} sh -c "install -Dm644 {} /usr/share/fonts/truetype/google-fonts" && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Cantarell-*.ttf" -delete && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Ubuntu-*.ttf" -delete && \
    apt-get --purge remove fonts-roboto && \
    fc-cache -f > /dev/null && \
    find fonts-${GOOGLE_FONTS_COMMIT} -name "METADATA.pb" | xargs -I{} bash -c "dirname {} | cut -d'/' -f3 | xargs printf; printf ","; grep -i 'subset' {} | cut -d':' -f2 | paste -sd "," - | sed 's/\"//g'" > /usr/share/fonts/truetype/google-fonts/google_fonts_metadata

# Install fonts from 1001fonts.com
# Some fonts fail to install because they contain ' (quotes) in their name. Skipping these for now.
RUN mkdir "1001-fonts" && \
    cd "1001-fonts" && \
    wget https://github.com/ElementAI/synbols-resources/releases/download/0.0.1/1001fonts.zip && \
    unzip 1001fonts.zip && \
    mkdir -p /usr/share/fonts/truetype/1001-fonts && \
    find . -type f -name "*.ttf" | sed "s/'//g" | xargs -I{} sh -c "install -Dm644 '{}' /usr/share/fonts/truetype/1001-fonts || true" && \
    fc-cache -f > /dev/null

# Summarize all google font licenses
RUN sh -c "echo \"license,font\"; find fonts-$GOOGLE_FONTS_COMMIT -name \"METADATA.pb\" | xargs -I{} bash -c \"dirname {} | cut -d'/' -f2,3 | sed -r 's/[/]+/,/g'\"" > font_licenses.csv

# ------------------------------------------------------------------------
# Synbols-dependent precomputation
# ------------------------------------------------------------------------
RUN mkdir bootstrap
COPY ./synbols ./bootstrap/synbols

# Package locales
COPY developer_tools ./bootstrap
RUN cd bootstrap && \
    python package_locales.py && \
    mkdir /locales && \
    cp locale_* /locales

# Bootstrap Cairo caches
COPY dummy_dataset.py ./bootstrap
RUN cd bootstrap && python dummy_dataset.py

# Bootstrap Matplotlib caches
RUN python -c "import matplotlib.pyplot as plt"

# Clean up
RUN rm -r /bootstrap
# ------------------------------------------------------------------------
FROM base AS test
RUN pip install pytest flake8
