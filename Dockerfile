# to build the docker image, run:
# docker build -t gyrokan .

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y

RUN apt install -y \
    git \
    build-essential \
    wget \
    unzip \
    pkg-config \
    cmake \
    pip \
    sudo \
    g++ \
    ca-certificates \
    htop \
    nano \
    libgl1-mesa-glx \
    gedit

RUN pip install --upgrade pip

RUN pip install pytorch-lightning torchvision seaborn ipython tensorboard tensorboardx optuna pandas scikit-learn matplotlib numpy torchmetrics

RUN pip install optuna-integration[pytorch_lightning] plotly kaleido  optuna-dashboard

RUN apt install -y libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2

RUN python -c "import kaleido; kaleido.get_chrome_sync()"

RUN useradd -m developer 

RUN echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer

USER developer

WORKDIR /home