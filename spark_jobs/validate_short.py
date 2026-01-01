import sys, os, json
from datetime import datetime, timezone, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, max as spark_max

DATA_DIR, ARTIFACT_DIR = sys.argv[1], sys.argv[2]

spark = SparkSession.builder.appName("validate_short").getOrCreate()

in_path = os.path.join(DATA_DIR, "processed", "short", "latest.parquet")
df = spark.read.parquet(in_path)

# Freshness: expect latest 1h candle within last 6 hours
latest_ts = df.select(spark_max("Datetime").alias("max_ts")).collect()[0]["max_ts"]
latest_ts_py = latest_ts.to_pydatetime().replace(tzinfo=timezone.utc)
now_utc = datetime.now(timezone.utc)

freshness_ok = (now_utc - latest_ts_py) <= timedelta(hours=6)

counts = df.groupBy("symbol").agg(count("close").alias("n")).collect()
missing_rate_ok = all([c["n"] > 100 for c in counts])

metrics = {
    "freshness_ok": bool(freshness_ok),
    "missing_rate_ok": bool(missing_rate_ok),
    "last_mape": None
}

out_dir = os.path.join(ARTIFACT_DIR, "validate", "short")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f)

print("[validate_short] metrics:", metrics)