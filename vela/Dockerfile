# Base image
FROM nvcr.io/nvidia/pytorch:24.01-py3

USER root

# Update and install necessary system packages
RUN apt-get update -y && \
    apt-get install -y iproute2 && \
    apt-get upgrade -y git

# Set up working directory
WORKDIR /app
RUN mkdir -p /app/cache
RUN mkdir -p /app/model
COPY . .

RUN pip install --upgrade pip setuptools

# Flash-attention settings
RUN pip install ninja
RUN pip uninstall -y flash-attn
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE \
    MAX_JOBS=4
RUN pip install flash-attn==2.3.6 --no-build-isolation

RUN pip install -r requirements.txt


RUN chmod -R a+rwx /app/
RUN chmod -R a+rwx /app/cache/