import sys, os
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, avg, stddev, expr

DATA_DIR, ARTIFACT_DIR = sys.argv[1], sys.argv[2]

spark = SparkSession.builder.appName("features_long").getOrCreate()

in_path = os.path.join(DATA_DIR, "processed", "long", "latest.parquet")
df = spark.read.parquet(in_path)

# Example features (per symbol)
w20 = Window.partitionBy("symbol").orderBy("Datetime").rowsBetween(-19, 0)

feat = (
    df.withColumn("sma20", avg(col("close")).over(w20))
      .withColumn("rolling_vol20", stddev(col("close")).over(w20))
)

out_path = os.path.join(DATA_DIR, "processed", "long", "latest.parquet")
feat.write.mode("overwrite").parquet(out_path)
print("[features_long] wrote features to", out_path)