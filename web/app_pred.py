
import os
import math
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, render_template, request, send_from_directory, abort
from flask_cors import CORS

import pandas as pd
import numpy as np

# Optional deps; only import if available
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
except Exception:
    SARIMAXResults = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.abspath("./artifacts"))
CHART_DIR = os.environ.get("CHART_DIR", os.path.abspath("./static/chart"))
DEFAULT_TICKER = os.environ.get("DEFAULT_TICKER", "GC=F")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ------------------------------
# Utilities
# ------------------------------
def _ensure_dirs():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _postprocess_ohlc(o, h, l, c):
    # Enforce OHLC constraints: high >= all, low <= all
    vals = [v for v in [o, h, l, c] if not (v is None or (isinstance(v,float) and math.isnan(v)))]
    if not vals:
        return (np.nan, np.nan, np.nan, np.nan)
    hi = max(vals)
    lo = min(vals)
    # If provided h/l are missing, set them; otherwise clamp
    h = hi if (h is None or math.isnan(h)) else max(h, hi, o if not math.isnan(o) else -np.inf, c if not math.isnan(c) else -np.inf)
    l = lo if (l is None or math.isnan(l)) else min(l, lo, o if not math.isnan(o) else np.inf, c if not math.isnan(c) else np.inf)
    # Ensure open/close lie within [l,h]
    if not math.isnan(o): o = min(max(o, l), h)
    if not math.isnan(c): c = min(max(c, l), h)
    return (o, h, l, c)

def _download_recent(ticker, period="60d", interval="1d"):
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.rename(columns=str.title)  # ensure Open/High/Low/Close
            return df
    except Exception:
        return None
    return None

def _atr(df, n=14):
    # Average True Range for bounds around predicted close
    if df is None or len(df) < 2:
        return None
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=n, min_periods=2).mean().iloc[-1]
    return float(atr) if pd.notnull(atr) else None

# ------------------------------
# Pages
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------------------
# Static chart serving (Altair/Vega-Lite JSON already generated)
# ------------------------------
@app.route("/static/chart/<path:filename>")
def serve_chart(filename):
    # Serve files from CHART_DIR safely
    if not os.path.isdir(CHART_DIR):
        abort(404)
    try:
        return send_from_directory(CHART_DIR, filename, as_attachment=False)
    except Exception:
        abort(404)

# ------------------------------
# Market trend values (for progress bars)
# ------------------------------
@app.route("/api/market-trends")
def market_trends():
    # You can wire these to real metrics later; for now read env or defaults
    data = {
        "1H": _safe_float(os.environ.get("TREND_1H", 51), 51),
        "24H": _safe_float(os.environ.get("TREND_24H", 41), 41),
        "1W": _safe_float(os.environ.get("TREND_1W", 73), 73),
        "1M": _safe_float(os.environ.get("TREND_1M", 73), 73),
    }
    return jsonify(data)

# ------------------------------
# Prediction endpoint
# ------------------------------
def _predict_naive(ticker: str, horizon: int = 1, interval="1d"):
    df = _download_recent(ticker, period="90d", interval=interval)
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    last_close = float(last["Close"])
    last_open = float(last["Open"])
    # Simple drift: use SMA ratio over last window
    window = min(10, len(df))
    sma = df["Close"].tail(window).mean()
    ratio = (last_close / sma) if sma and sma != 0 else 1.0
    pred_close = last_close * (0.5 + 0.5 * ratio)  # soft blend towards mean
    # Bounds using ATR
    atr_val = _atr(df, n=14) or (last["High"] - last["Low"]) or abs(last_close * 0.01)
    pred_open = (pred_close + last_open) / 2.0
    pred_high = max(pred_close, pred_open) + 0.5 * atr_val
    pred_low = min(pred_close, pred_open) - 0.5 * atr_val
    o, h, l, c = _postprocess_ohlc(pred_open, pred_high, pred_low, pred_close)
    ts = (df.index[-1] + pd.Timedelta(days=horizon)).to_pydatetime().astimezone()
    return {
        "ticker": ticker,
        "timestamp": ts.isoformat(timespec="seconds"),
        "open": round(float(o), 4),
        "high": round(float(h), 4),
        "low": round(float(l), 4),
        "close": round(float(c), 4),
        "model_used": "naive_sma_atr",
        "note": "Fallback forecast using SMA drift and ATR bounds."
    }

@app.route("/api/predict")
def api_predict():
    ticker = request.args.get("ticker", DEFAULT_TICKER)
    horizon = int(request.args.get("horizon", 1))
    interval = request.args.get("interval", "1d")
    # TODO: attempt to load your trained models from ARTIFACTS_DIR
    # This sample uses a robust naive fallback; plug in SARIMAX/Prophet/LSTM below if available.
    result = _predict_naive(ticker, horizon=horizon, interval=interval)
    if result is None:
        return jsonify({"error": "Unable to compute prediction (no data)."}), 500
    return jsonify(result)

# ------------------------------
# Health
# ------------------------------
@app.route("/healthz")
def healthz():
    _ensure_dirs()
    return jsonify({"status": "ok", "time": _now_iso(), "artifacts_dir": ARTIFACTS_DIR, "chart_dir": CHART_DIR})

if __name__ == "__main__":
    _ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
