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
COPY docker/requirements.txt .
RUN pip install -r requirements.txt

# Download Natural images (CC License)
RUN mkdir -p /images && cd /images && \
    wget https://github.com/ElementAI/synbols-resources/raw/master/datasets/natural_images.tar.gz && \
    tar -xf natural_images.tar.gz && \
    rm natural_images.tar.gz

# Install all Google fonts and extract their metadata
RUN wget -O google_fonts.zip https://github.com/google/fonts/archive/${GOOGLE_FONTS_COMMIT}.zip && \
    unzip google_fonts.zip && \
    mkdir -p /usr/share/fonts/truetype/google-fonts && \
    find fonts-${GOOGLE_FONTS_COMMIT} -type f -name "*.ttf" | xargs -I{} sh -c "install -Dm644 {} /usr/share/fonts/truetype/google-fonts" && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Cantarell-*.ttf" -delete && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Ubuntu-*.ttf" -delete && \
    apt-get --purge remove fonts-roboto && \
    fc-cache -f > /dev/null && \
    find fonts-${GOOGLE_FONTS_COMMIT} -name "METADATA.pb" | xargs -I{} bash -c "dirname {} | cut -d'/' -f3 | xargs printf; printf ","; grep -i 'subset' {} | cut -d':' -f2 | paste -sd "," - | sed 's/\"//g'" > /usr/share/fonts/truetype/google-fonts/google_fonts_metadata && \
    sh -c "echo \"license,font\"; find fonts-$GOOGLE_FONTS_COMMIT -name \"METADATA.pb\" | xargs -I{} bash -c \"dirname {} | cut -d'/' -f2,3 | sed -r 's/[/]+/,/g'\"" > font_licenses.csv && \
    rm -r fonts-${GOOGLE_FONTS_COMMIT} google_fonts.zip

# Install fonts from 1001freefonts.com
# Some fonts fail to install because they contain ' (quotes) in their name. Skipping these for now.
ENV A=1
RUN mkdir "1001-fonts" && \
    cd "1001-fonts" && \
    wget https://github.com/ElementAI/synbols-resources/releases/download/0.0.1/1001fonts.zip && \
    unzip 1001fonts.zip && \
    mkdir -p /usr/share/fonts/truetype/1001-fonts && \
    find . -type f -name "*.ttf" | sed "s/'//g" | xargs -I{} sh -c "install -Dm644 '{}' /usr/share/fonts/truetype/1001-fonts || true" && \
    fc-cache -f > /dev/null && \
    cd .. && rm -r 1001-fonts

# Install extra asian fonts
RUN mkdir "asian_fonts" && \
    cd "asian_fonts" && \
    wget https://github.com/ElementAI/synbols-resources/releases/download/0.0.1/asian_fonts.zip && \
    unzip asian_fonts.zip && \
    mkdir -p /usr/share/fonts/truetype/asian_fonts && \
    find . -type f -name "*.ttf" | sed "s/'//g" | xargs -I{} sh -c "install -Dm644 '{}' /usr/share/fonts/truetype/asian_fonts || true" && \
    fc-cache -f > /dev/null && \
    cd .. && rm -r asian_fonts

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
COPY docker/dummy_dataset.py ./bootstrap
RUN cd bootstrap && python dummy_dataset.py

# Bootstrap Matplotlib caches
RUN python -c "import matplotlib.pyplot as plt"

# Clean up
RUN rm -r /bootstrap
# ------------------------------------------------------------------------



# ------------------------------------------------------------------------
# Notebook examples
# ------------------------------------------------------------------------
RUN pip install jupyter matplotlib pytorch_lightning torch torchvision
RUN pip install --upgrade ipython Pygments>=2.6.1
RUN mkdir /.local && chmod -R 777 /.local

# Copy examples to image
COPY ./examples/ /examples
# ------------------------------------------------------------------------


FROM base AS test
RUN pip install pytest flake8

FROM base AS sphinx
ADD ./docs/requirements.txt docs/requirements.txt
RUN pip install -r docs/requirements.txt

