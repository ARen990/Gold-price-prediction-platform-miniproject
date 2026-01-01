FROM bitnami/spark:3.5
USER root
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir pandas yfinance numpy
USER 1001