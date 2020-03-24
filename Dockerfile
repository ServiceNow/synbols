FROM ubuntu:18.04
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
    apt-get install -y fonts-cantarell fontconfig git icu-devtools ipython3 libcairo2-dev libhdf5-dev pkg-config ttf-ubuntu-font-family unzip wget

# Install all python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

## Install all Google fonts and extract their metadata
RUN wget -O google_fonts.zip https://github.com/google/fonts/archive/${GOOGLE_FONTS_COMMIT}.zip && \
    unzip google_fonts.zip && \
    mkdir -p /usr/share/fonts/truetype/google-fonts && \
    find fonts-${GOOGLE_FONTS_COMMIT} -type f -name "*.ttf" | xargs -I{} sh -c "install -Dm644 {} /usr/share/fonts/truetype/google-fonts" && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Cantarell-*.ttf" -delete && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Ubuntu-*.ttf" -delete && \
    apt-get --purge remove fonts-roboto && \
    fc-cache -f > /dev/null && \
    find fonts-${GOOGLE_FONTS_COMMIT} -name "METADATA.pb" | xargs -I{} bash -c "dirname {} | cut -d'/' -f3 | xargs printf; printf ","; grep -i 'subset' {} | cut -d':' -f2 | paste -sd "," - | sed 's/\"//g'" > /usr/share/fonts/truetype/google-fonts/google_fonts_metadata

ENV PYTHONPATH "${PYTHONPATH}:/synbols"