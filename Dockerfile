FROM ubuntu:18.04
#MAINTAINER TODO

# Install Python 3
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    pip3 install --upgrade pip

# Install system dependencies
RUN apt-get install -y fonts-cantarell fontconfig git libcairo2-dev pkg-config ttf-ubuntu-font-family unzip wget

# Install all python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install all Google fonts and extract their metadata
RUN wget https://github.com/google/fonts/archive/master.zip && \
    unzip master.zip && \
    mkdir -p /usr/share/fonts/truetype/google-fonts && \
    find fonts-master -type f -name "*.ttf" | xargs -I{} sh -c "install -Dm644 {} /usr/share/fonts/truetype/google-fonts" && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Cantarell-*.ttf" -delete && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Ubuntu-*.ttf" -delete && \
    apt-get --purge remove fonts-roboto && \
    fc-cache -f > /dev/null && \
    find fonts-master -name "METADATA.pb" | xargs -I{} bash -c "dirname {} | cut -d'/' -f3 | xargs printf; printf ","; grep -i 'subset' {} | cut -d':' -f2 | paste -sd "," - | sed 's/\"//g'" > /usr/share/fonts/truetype/google-fonts/google_fonts_metadata