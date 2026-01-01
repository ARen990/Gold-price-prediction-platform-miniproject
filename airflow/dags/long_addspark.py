from __future__ import annotations
import json, os
from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.trigger_rule import TriggerRule

TZ = pendulum.timezone("Asia/Bangkok")

SPARK_APP = "/opt/spark-apps/etl_long.py"           # จาก compose: ./spark_jobs:/opt/spark-apps
METRIC_DIR = "/opt/artifacts/validate"              # directory ที่เก็บ metrics.json
METRIC_FILE = "/opt/artifacts/validate/metrics.json" # path เต็มไปยังไฟล์ metrics.json
JOBS_DIR = "/opt/jobs"                               # จาก compose: ./jobs:/opt/jobs

default_args = {
    "owner": "aie",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="long_rubric",
    start_date=TZ.convert(pendulum.datetime(2025, 10, 1, 0, 0)),
    schedule_interval="0 12 * * *",  # 12:00 Asia/Bangkok
    catchup=False,
    default_args=default_args,
    tags=["spark","rubric","gold","long"],
) as dag:

    # 1) Start/Stop (รูบิกข้อ 1)
    start = EmptyOperator(task_id="start")
    stop  = EmptyOperator(task_id="stop", trigger_rule=TriggerRule.ALL_DONE)

    # 2) PySpark Processing (รูบิกข้อ 2)
    spark_etl = SparkSubmitOperator(
        task_id="spark_etl",
        application=SPARK_APP,
        conn_id="spark_default",
        name="etl_long_job",
        application_args=[METRIC_DIR],  # ส่ง directory เป็น argument
        conf={
        # 👇 บอกให้ executor/worker ติดต่อกลับมาที่ scheduler
        "spark.driver.host": "airflow-scheduler",
        "spark.driver.bindAddress": "0.0.0.0",
        # (ออปชัน) กัน DNS หลุด:
        "spark.executor.instances": "1",
        },
        env_vars={
        # รายการสัญลักษณ์คั่นด้วย comma
        "SYMBOLS": "GC=F,BTC-USD,^GSPC,SLV,EURUSD=X,^DJI,CL=F",
        # กรองข้อมูลไม่เกินวันที่นี้ (ออปชัน ไม่ใส่ก็ได้)
        "END_DATE": "2025-10-03",
        # พาธผลลัพธ์ (ตรงกับ compose ของคุณ)
        "OUT_PARQUET": "/opt/shared-data/processed/long/latest.parquet",
        "METRIC_JSON": "/opt/artifacts/validate/metrics.json",
        # เกณฑ์คัดแถว: จำนวนคอลัมน์ที่ยอมให้ว่างได้ต่อแถว
        "MISSING_THRESHOLD": "15",
        },
        verbose=True,
    )

    # ดึง metric จากไฟล์ที่ Spark เขียน แล้ว push เป็น XCom
    def read_metrics_to_xcom(**context):
        # ใช้ METRIC_FILE แทน METRIC_DIR
        metric_path = METRIC_FILE
        
        # ตรวจสอบว่าไฟล์มีอยู่จริง
        if not os.path.exists(metric_path):
            raise FileNotFoundError(f"Metrics file not found: {metric_path}")
        
        print(f"Reading metrics from: {metric_path}")
        
        with open(metric_path, "r") as f:
            data = json.load(f)
        
        print(f"Metrics data: {data}")
        
        missing_rate = data.get("missing_rate", 1.0)
        context["ti"].xcom_push(key="missing_rate", value=missing_rate)
        
        # Push ข้อมูลทั้งหมดไปยัง XCom ด้วย
        context["ti"].xcom_push(key="metrics_data", value=data)

    read_metric = PythonOperator(
        task_id="read_metric",
        python_callable=read_metrics_to_xcom,
        provide_context=True,
    )

    # 3) Branching/Decision (รูบิกข้อ 3)
    def decide_branch(**context):
        rate = context["ti"].xcom_pull(task_ids="read_metric", key="missing_rate")
        print(f"Missing rate from XCom: {rate}")
        
        if rate is None:
            print("No missing rate found in XCom, going to fail_path")
            return "fail_path"
        
        # ใช้ threshold 10% ตาม requirement เดิม
        if rate <= 0.10:
            print(f"Missing rate {rate} <= 0.10, going to ok_path")
            return "ok_path"
        else:
            print(f"Missing rate {rate} > 0.10, going to fail_path")
            return "fail_path"

    branch = BranchPythonOperator(
        task_id="branch_by_quality",
        python_callable=decide_branch,
        provide_context=True,
    )

    ok_path   = EmptyOperator(task_id="ok_path")
    fail_path = EmptyOperator(task_id="fail_path")

    # งานเดิม (train/predict) ใช้สคริปต์ที่คุณให้มา (ไม่แก้ logic ภายใน)
    task_train = BashOperator(
        task_id="task_train",
        bash_command=f"python {JOBS_DIR}/long/train_long.py",
    )
    task_predict = BashOperator(
        task_id="task_predict",
        bash_command=f"python {JOBS_DIR}/long/predict_long.py",
    )

    # ลำดับงาน (รวม XCom แล้วคือรูบิกข้อ 4)
    start >> spark_etl >> read_metric >> branch
    branch >> ok_path >> task_train >> task_predict >> stop
    branch >> fail_path >> stop