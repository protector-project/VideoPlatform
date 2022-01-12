# PROTECTOR VIDEO PROCESSING

[//]: # (This is a comment, it will not be included)

## Requirements

We highly recommend the use of GPU to processing video. We recommend install the NVIDIA tool:

```
sudo apt install -y docker.io nvidia-container-toolkit
```

Restart docker service

```
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## Build Image

```
docker build --pulll --rm -f "Dockerfile" -t protectorvideo:latest "."
```

## Run Image

```
docker run --gpus all -it --rm -v `pwd`/data/:/data/ protectorvideo:latest
```

## Run processing

```
python main.py -I <video> -N <video_place> -AM <anomaly-model>
```
