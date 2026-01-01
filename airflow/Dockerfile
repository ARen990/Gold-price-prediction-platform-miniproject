FROM apache/airflow:2.10.2-python3.11

USER root

# ── System deps
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    tzdata \
    procps \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Spark client (ตรงกับ bitnami/spark:3.5)
RUN curl -fsSL -o /tmp/spark.tgz https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz \
 && tar -xzf /tmp/spark.tgz -C /opt \
 && mv /opt/spark-3.5.1-bin-hadoop3 /opt/spark \
 && rm /tmp/spark.tgz

# ── Env
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH="$SPARK_HOME/bin:$PATH"
ENV AIRFLOW__CORE__DEFAULT_TIMEZONE="Asia/Bangkok"

# (ทางเลือก) เผื่อคุณวางสคริปต์ฝั่ง train/predict ไว้ที่ /opt/jobs
ENV PYTHONPATH="/opt/jobs:${PYTHONPATH}"

# ── Python deps for driver (etl_long.py ใช้ yfinance/pandas ก่อนเข้า Spark)
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir \
      -r /requirements.txt \
      -c https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-3.11.txt
