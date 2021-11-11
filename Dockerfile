FROM nvcr.io/nvidia/pytorch:21.06-py3

COPY . /app/

WORKDIR /app

RUN pip3 install -r requirements.txt