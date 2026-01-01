import argparse
from pathlib import Path
import sys
import pandas as pd
import yfinance as yf
import numpy as np

DEFAULT_SYMBOLS = ["GC=F"]  # add more: "BTC-USD","^GSPC","^DJI","CL=F","SI=F","EURUSD=X"

def _download_block(symbols, period, interval, auto_adjust=False):
    frames = []
    for sym in symbols:
        df_sym = yf.download(
            sym,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
        )
        if df_sym.empty:
            print(f"[WARN] No data for {sym}")
            continue
        df_sym = df_sym[["Open", "High", "Low", "Close"]].copy()
        # make columns unique per symbol
        df_sym.columns = [f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close"]
        frames.append(df_sym)

    if not frames:
        print("[FATAL] No data downloaded.", file=sys.stderr)
        sys.exit(2)

    df = pd.concat(frames, axis=1).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def clean_dataframe(df):
    """
    ทำความสะอาด DataFrame - ลบบรรทัดที่มีค่า null, NaN, หรือ blank ทั้งหมด
    """
    print(f"Cleaning data: {len(df)} rows before cleaning")
    
    # สร้างสำเนาเพื่อป้องกันการแก้ไขข้อมูลต้นฉบับ
    df_clean = df.copy()
    
    # ตรวจจับและลบบรรทัดที่มีปัญหา
    rows_before = len(df_clean)
    
    # 1. ลบบรรทัดที่มีค่า NaN ในทุกคอลัมน์
    nan_mask = df_clean.isna().all(axis=1)
    df_clean = df_clean[~nan_mask]
    nan_removed = nan_mask.sum()
    
    # 2. ลบบรรทัดที่มีค่า Inf/-Inf
    inf_mask = np.isinf(df_clean).any(axis=1)
    df_clean = df_clean[~inf_mask]
    inf_removed = inf_mask.sum()
    
    # 3. ลบบรรทัดที่มีค่า null (None)
    null_mask = df_clean.isnull().all(axis=1)
    df_clean = df_clean[~null_mask]
    null_removed = null_mask.sum()
    
    # 4. ลบบรรทัดที่มีค่า 0 ในทุกคอลัมน์ (blank)
    zero_mask = (df_clean == 0).all(axis=1)
    df_clean = df_clean[~zero_mask]
    zero_removed = zero_mask.sum()
    
    # 5. ลบบรรทัดที่มีค่าผิดปกติ (เช่น ราคาติดลบ)
    abnormal_mask = False
    for col in df_clean.columns:
        if 'open' in col.lower() or 'high' in col.lower() or 'low' in col.lower() or 'close' in col.lower():
            # ตรวจสอบว่าราคาเป็นค่าบวกและไม่ใช่ค่าผิดปกติ
            abnormal_mask = abnormal_mask | (df_clean[col] <= 0) | (df_clean[col] > 100000)  # ราคาทองไม่น่าจะเกิน 100,000
    
    abnormal_removed = abnormal_mask.sum() if isinstance(abnormal_mask, pd.Series) else 0
    df_clean = df_clean[~abnormal_mask]
    
    rows_after = len(df_clean)
    total_removed = rows_before - rows_after
    
    print(f"Cleaning results:")
    print(f"   Rows before cleaning: {rows_before}")
    print(f"   Rows after cleaning: {rows_after}")
    print(f"   Total rows removed: {total_removed}")
    print(f"   Breakdown:")
    print(f"     - NaN rows: {nan_removed}")
    print(f"     - Inf rows: {inf_removed}")
    print(f"     - Null rows: {null_removed}")
    print(f"     - Zero rows: {zero_removed}")
    print(f"     - Abnormal rows: {abnormal_removed}")
    
    # ตรวจสอบข้อมูลหลังทำความสะอาด
    if rows_after > 0:
        print(f"Data quality after cleaning:")
        print(f"   - Missing values: {df_clean.isna().sum().sum()}")
        print(f"   - Infinite values: {np.isinf(df_clean).sum().sum()}")
        print(f"   - Zero values: {(df_clean == 0).sum().sum()}")
    else:
        print("WARNING: No valid data remaining after cleaning!")
    
    return df_clean

def build_short(symbols):
    """
    SHORT horizon:
      - period=30d, interval=1h
      - keep rows with <=60% missing
      - linear interpolate + ffill + bfill; fill remaining NaN with 0
    """
    print(">> Downloading latest market data (SHORT: 30d @1h)…")
    df_raw = _download_block(symbols, period="30d", interval="1h", auto_adjust=False)
    df = df_raw.copy()

    # lenient cleaning for hourly data (matches your snippet, but fixed comment to 60%)
    missing_pct = df.isna().sum(axis=1) / len(df.columns) * 100.0
    df = df[missing_pct <= 60]

    if df.empty:
        print("[WARN] All rows filtered out. Using original data with forward fill.")
        df = df_raw.copy()

    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0)
    
    df_clean = clean_dataframe(df)
    
    print(f"✓ SHORT cleaned rows: {len(df_clean)} (from {len(df)})")
    return df_clean

def build_long(symbols):
    """
    LONG horizon:
      - period=5y, interval=1d
      - drop rows with >100 NaNs across all symbol columns
      - linear interpolate + ffill + bfill
    """
    print(">> Downloading latest market data (LONG: 5y @1d)…")
    df = _download_block(symbols, period="5y", interval="1d", auto_adjust=False)

    # strict-ish daily cleaning
    df = df[df.isna().sum(axis=1) <= 100]
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()
    
    df_clean = clean_dataframe(df)
    
    print(f"✓ LONG cleaned rows: {len(df_clean)} (from {len(df)})")
    return df_clean

def _write_csv(df, outdir: Path, filename: str, split_by_symbol: bool):
    # Create directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing file if it exists
    path = outdir / filename
    if path.exists():
        print(f"Removing existing file: {path}")
        path.unlink()
    
    df_out = df.copy()
    df_out.index.name = "Date"
    
    print(f"Final data check before writing {filename}:")
    print(f"   - Total rows: {len(df_out)}")
    print(f"   - Total columns: {len(df_out.columns)}")
    print(f"   - Missing values: {df_out.isna().sum().sum()}")
    print(f"   - Infinite values: {np.isinf(df_out).sum().sum()}")
    
    # Write new file
    df_out.to_csv(path)
    print(f"-> Wrote {path.resolve()} (rows={len(df_out)}, cols={len(df_out.columns)})")

    if split_by_symbol:
        # write per-symbol CSVs with canonical OHLCV column names
        # input columns look like: "<SYM>_open" ... "<SYM>_volume"
        by_sym = {}
        for c in df_out.columns:
            if "_" not in c:
                continue
            sym, feat = c.split("_", 1)
            by_sym.setdefault(sym, {})[feat] = c

        per_dir = outdir / (path.stem + "_by_symbol")
        per_dir.mkdir(parents=True, exist_ok=True)

        for sym, mapping in by_sym.items():
            cols = [mapping.get(k) for k in ["open", "high", "low", "close"] if mapping.get(k)]
            sdf = df_out[cols].rename(columns={
                mapping.get("open", ""): "Open",
                mapping.get("high", ""): "High",
                mapping.get("low", ""): "Low",
                mapping.get("close", ""): "Close",
            })
            out = per_dir / f"{sym}.csv"
            # Remove existing per-symbol file if it exists
            if out.exists():
                out.unlink()
            sdf.to_csv(out, index=True)
            print(f"   -> {out.name} (rows={len(sdf)})")

def main():
    # Calculate default output directory: project/web/static/chart
    script_dir = Path(__file__).parent.resolve()
    default_outdir = script_dir.parent / "web" / "static" / "chart"
    
    ap = argparse.ArgumentParser(description="Pull cleaned OHLCV to CSV (SHORT 30d@1h, LONG 5y@1d).")
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                    help="Comma-separated symbols, e.g. 'GC=F,BTC-USD,CL=F,SI=F'")
    ap.add_argument("--horizon", choices=["short", "long", "both"], default="both",
                    help="Which horizon to build.")
    ap.add_argument("--outdir", type=str, default=str(default_outdir),
                    help="Output directory for CSV files.")
    ap.add_argument("--split-by-symbol", action="store_true",
                    help="Also write per-symbol CSVs in a subfolder.")
    ap.add_argument("--strict-cleaning", action="store_true",
                    help="Enable strict data cleaning (remove all problematic rows)")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    outdir = Path(args.outdir)

    print(f"Output directory: {outdir.resolve()}")

    if args.horizon in ("short", "both"):
        print("\n" + "="*50)
        print("BUILDING SHORT-TERM DATA (30d @1h)")
        print("="*50)
        short_df = build_short(symbols)
        _write_csv(short_df, outdir, "short_historic.csv", args.split_by_symbol)

    if args.horizon in ("long", "both"):
        print("\n" + "="*50)
        print("BUILDING LONG-TERM DATA (5y @1d)")
        print("="*50)
        long_df = build_long(symbols)
        _write_csv(long_df, outdir, "long_historic.csv", args.split_by_symbol)

if __name__ == "__main__":
    main()