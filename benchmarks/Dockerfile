FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
COPY requirements.txt /app/synbols/requirements.txt
RUN pip install -r /app/synbols/requirements.txt
COPY . /app/synbols
WORKDIR /app/synbols
