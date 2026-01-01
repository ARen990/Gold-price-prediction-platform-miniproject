import sys, os
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, avg, stddev

DATA_DIR, ARTIFACT_DIR = sys.argv[1], sys.argv[2]

spark = SparkSession.builder.appName("features_short").getOrCreate()

in_path = os.path.join(DATA_DIR, "processed", "short", "latest.parquet")
df = spark.read.parquet(in_path)

w20 = Window.partitionBy("symbol").orderBy("Datetime").rowsBetween(-19, 0)
feat = (
    df.withColumn("sma20", avg(col("close")).over(w20))
      .withColumn("rolling_vol20", stddev(col("close")).over(w20))
)

out_path = os.path.join(DATA_DIR, "processed", "short", "latest.parquet")
feat.write.mode("overwrite").parquet(out_path)
print("[features_short] wrote features to", out_path)