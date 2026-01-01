# spark_jobs/etl_short.py
import os, sys, json
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

def get_env(name: str, default: str = ""):
    return os.environ.get(name, default)

def download_one(sym: str) -> pd.DataFrame:
    df = yf.download(sym, period="730d", interval="1h", auto_adjust=False, progress=False)
    if df.empty:
        return df
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = [f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close", f"{sym}_volume"]
    # (ตัวอย่าง indicator เบาๆ ใน pandas แล้วส่งเข้า Spark)
    df[f"{sym}_sma14"] = df[f"{sym}_close"].rolling(14, min_periods=1).mean()
    df[f"{sym}_ema14"] = df[f"{sym}_close"].ewm(span=14, adjust=False).mean()
    # RSI แบบเร็วๆ
    delta = df[f"{sym}_close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(14, min_periods=1).mean()
    rs = gain / (loss.replace(0, pd.NA))
    df[f"{sym}_rsi14"] = 100 - (100 / (1 + rs))
    return df

def merge_on_index(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    # รวมตาม index (Datetime) แล้ว sort
    out = pd.concat(dfs, axis=1).sort_index()
    # บังคับเป็น numeric เพื่อให้ Spark แปลงได้ดี
    return out.apply(pd.to_numeric, errors="coerce")

def spark_ffill_bfill(sdf):
    # เติมค่าแบบ ffill/bfill สำหรับทุกคอลัมน์ตัวเลข
    num_cols = [c for (c, t) in sdf.dtypes if t in ("double", "float", "int", "bigint")]
    # Forward fill
    w = Window.orderBy("datetime").rowsBetween(Window.unboundedPreceding, 0)
    for c in num_cols:
        sdf = sdf.withColumn(f"{c}__ffill", F.last(F.col(c), ignorenulls=True).over(w))
    # Backward fill
    w_rev = Window.orderBy(F.col("datetime").desc()).rowsBetween(Window.unboundedPreceding, 0)
    for c in num_cols:
        sdf = sdf.withColumn(f"{c}__bfill", F.last(F.col(c), ignorenulls=True).over(w_rev))
    # coalesce(ffill, bfill, original)
    for c in num_cols:
        sdf = sdf.withColumn(c, F.coalesce(F.col(f"{c}__ffill"), F.col(f"{c}__bfill"), F.col(c)))
        sdf = sdf.drop(f"{c}__ffill", f"{c}__bfill")
    return sdf

def count_missing_per_row(sdf):
    # นับจำนวนคอลัมน์ที่เป็น null ต่อแถว
    cols = [F.col(c).isNull().cast("int") for c in sdf.columns if c != "datetime"]
    return sdf.withColumn("_missing_cnt", sum(cols))

def main(out_json_path: str | None = None):
    symbols = ["GC=F", "BTC-USD", "^GSPC", "SLV", "EURUSD=X", "^DJI", "CL=F"]
    # END_DATE can be provided via environment; default to today's UTC date
    end_date_env = get_env("END_DATE", "")
    if end_date_env:
        end_date = pd.to_datetime(end_date_env)
        print(f"Using END_DATE from environment: {end_date.date()}")
    else:
        end_date = pd.to_datetime(datetime.utcnow().date())
        print(f"No END_DATE provided, defaulting to today's date (UTC): {end_date.date()}")
    parquet_out = get_env("OUT_PARQUET", "/opt/shared-data/processed/short/latest.parquet")
    
    # จัดการ path ให้ถูกต้อง
    if out_json_path:
        if os.path.isdir(out_json_path):
            metric_json = os.path.join(out_json_path, "metrics.json")
        else:
            metric_json = out_json_path
    else:
        metric_json = get_env("METRIC_JSON", "/opt/artifacts/validate/metrics.json")

    # ใช้ 20% แทน fixed threshold
    miss_th_percent = 0.20

    # 1) ดึงข้อมูล
    frames = []
    for sym in symbols:
        print(f"Downloading {sym}...")
        pdf = download_one(sym)
        if not pdf.empty:
            frames.append(pdf)
            print(f"  {sym}: {len(pdf)} rows")
        else:
            print(f"  {sym}: NO DATA")
    
    if not frames:
        raise SystemExit("[FATAL] No data downloaded from yfinance.")
    
    pdf_all = merge_on_index(frames)
    original_rows = len(pdf_all)

    # Reset index - แก้ไขส่วนนี้
    pdf_all = pdf_all.reset_index()
    
    # Debug: ดูชื่อคอลัมน์ที่มีอยู่
    print("Columns after reset_index:", pdf_all.columns.tolist())
    
    # หาชื่อคอลัมน์ datetime ที่ถูกต้อง
    datetime_cols = [col for col in pdf_all.columns if col.lower() in ['datetime', 'date', 'index']]
    if not datetime_cols:
        raise SystemExit("[FATAL] No datetime column found after merge.")
    
    # ใช้คอลัมน์ datetime แรกที่พบ
    datetime_col = datetime_cols[0]
    print(f"Using datetime column: '{datetime_col}'")
    
    # เปลี่ยนชื่อเป็น 'datetime' เสมอ
    if datetime_col != 'datetime':
        pdf_all = pdf_all.rename(columns={datetime_col: 'datetime'})

    # แปลงเป็น datetime และจัดการ timezone
    pdf_all["datetime"] = pd.to_datetime(pdf_all["datetime"])
    
    # หาก datetime มี timezone ให้แปลงทั้งหมดเป็น timezone เดียวกัน (UTC)
    if pdf_all["datetime"].dt.tz is not None:
        print("Datetime has timezone, converting all to UTC...")
        pdf_all["datetime"] = pdf_all["datetime"].dt.tz_convert('UTC')
    else:
        # หากไม่มี timezone ให้ตั้งเป็น UTC
        print("Datetime has no timezone, assuming UTC...")
        pdf_all["datetime"] = pdf_all["datetime"].dt.tz_localize('UTC')

    # 3) ทำความสะอาดข้อมูล
    # 3.1 กรองตาม end_date - แก้ไขส่วนนี้เพื่อจัดการ timezone
    if end_date_env:
        end_date = pd.to_datetime(end_date_env)
        # ตั้ง timezone ให้ตรงกับข้อมูล (UTC)
        end_date = end_date.tz_localize('UTC')
        print(f"Filtering data before: {end_date}")
        pdf_all = pdf_all[pdf_all["datetime"] <= end_date]

    # 3.2 กรองแถวที่มี missing มากกว่า 20%
    data_columns = [col for col in pdf_all.columns if col != "datetime"]
    total_data_columns = len(data_columns)
    missing_per_row = pdf_all[data_columns].isnull().sum(axis=1)
    missing_ratio = missing_per_row / total_data_columns
    
    print(f"Total data columns: {total_data_columns}")
    print(f"Applying 20% missing threshold...")
    print(f"Rows before filtering: {len(pdf_all)}")
    
    pdf_all = pdf_all[missing_ratio <= miss_th_percent]
    
    print(f"Rows after 20% filtering: {len(pdf_all)}")

    # 3.3 เติมค่าด้วย ffill + bfill
    pdf_all = pdf_all.sort_values("datetime")
    pdf_all[data_columns] = pdf_all[data_columns].ffill().bfill()
    
    after_rows = len(pdf_all)

    # 4) บันทึกผลลัพธ์
    out_dir = os.path.dirname(parquet_out)
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.exists(parquet_out) and os.path.isdir(parquet_out):
        import shutil
        shutil.rmtree(parquet_out)
    
    print(f"Saving to: {parquet_out}")
    
    # ก่อนบันทึกให้แปลง timezone เป็น naive (เอา timezone ออก) เพื่อความเข้ากันได้
    pdf_all["datetime"] = pdf_all["datetime"].dt.tz_localize(None)
    pdf_all.to_parquet(parquet_out, index=False)

    # 5) เขียน metrics
    missing_rate = 0.0 if after_rows == 0 else max(0.0, (original_rows - after_rows) / float(original_rows))
    os.makedirs(os.path.dirname(metric_json), exist_ok=True)
    with open(metric_json, "w") as f:
        json.dump({
            "symbols": symbols,
            "original_rows": int(original_rows),
            "after_cleaning_rows": int(after_rows),
            "missing_rate": float(missing_rate),
            "missing_threshold_percent": 20.0,
            "parquet_out": parquet_out,
            "ended_at": datetime.utcnow().isoformat() + "Z"
        }, f)

    print("ETL process completed successfully!")
    print(f"Final result: {after_rows} rows (kept {100*(1-missing_rate):.1f}% of original data)")
    
if __name__ == "__main__":
    arg_out = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg_out)