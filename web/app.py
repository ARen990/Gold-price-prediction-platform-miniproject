import os
import time
import traceback
import json, threading
import yfinance as yf
import pandas as pd

from datetime import datetime
from pathlib import Path
from flask import (
    Flask, render_template, send_file, jsonify, abort,
    make_response, send_from_directory, request
)
from flask_cors import CORS
from threading import Event
_bootstrap_started = Event()

# Your chart builders
from convert_path import generate_forecast_chart
from convert_path_all import generate_all_jsons

# ---------------- App & Paths ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # enable if you need cross-origin access

_primed_lock = threading.Lock()
app.config.setdefault("_HISTORY_PRIMED", False)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(BASE_DIR, "forecasts")
# canonical artifacts forecasts folder (sibling of web)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
FORECAST_ARTIFACTS_DIR = os.path.join(PROJECT_DIR, 'artifacts', 'forecasts')
HIST_BASE_DIR = os.path.join(BASE_DIR, 'static', 'chart')
JSON_DIR = os.path.join(BASE_DIR, "static", "chart")
os.makedirs(JSON_DIR, exist_ok=True)

DEFAULT_SYMBOLS = os.environ.get("PULL_SYMBOLS", "GC=F").split(",")  # e.g. "GC=F,BTC-USD,CL=F"
STATIC_CHART = os.path.join(BASE_DIR, "static", "chart")
os.makedirs(STATIC_CHART, exist_ok=True)

# Try to (re)build ALL-model JSONs at startup using artifact CSVs; don't fail hard if history missing
try:
    # use the artifacts forecasts folder as the historical base so generated CSVs are found
    generate_all_jsons(
        out_dir=STATIC_CHART,
        hist_base_dir=FORECAST_ARTIFACTS_DIR,
        short_inputs={
            'Prophet':      os.path.join(FORECAST_ARTIFACTS_DIR, 'short', 'short_forecast_prophet.csv'),
            'SARIMAX':      os.path.join(FORECAST_ARTIFACTS_DIR, 'short', 'short_forecast_sarimax.csv'),
            'LSTM':         os.path.join(FORECAST_ARTIFACTS_DIR, 'short', 'short_forecast_lstm.csv'),
            'Bayesian VAR': os.path.join(FORECAST_ARTIFACTS_DIR, 'short', 'short_forecast_bayesian_var.csv'),
        },
        long_inputs={
            'Prophet':      os.path.join(FORECAST_ARTIFACTS_DIR, 'long', 'long_forecast_prophet.csv'),
            'SARIMAX':      os.path.join(FORECAST_ARTIFACTS_DIR, 'long', 'long_forecast_sarimax.csv'),
            'LSTM':         os.path.join(FORECAST_ARTIFACTS_DIR, 'long', 'long_forecast_lstm.csv'),
            'Bayesian VAR': os.path.join(FORECAST_ARTIFACTS_DIR, 'long', 'long_forecast_bayesian_var.csv'),
        },
    )
except Exception as e:
    # Emit a warning but allow the Flask app to start; missing history is common in fresh checkouts
    print(f"[WARN] generate_all_jsons skipped at startup: {e}")

# Periods treated as "short" on the backend
SHORT_PERIODS = {"1H", "24H", "1W"}

# ---- CSV maps for ALL (edit if your filenames differ) ----
ALL_INPUTS_LONG = {
    "Prophet":      os.path.join(FORECAST_ARTIFACTS_DIR, "long",  "long_forecast_prophet.csv"),
    "SARIMAX":      os.path.join(FORECAST_ARTIFACTS_DIR, "long",  "long_forecast_sarimax.csv"),
    "LSTM":         os.path.join(FORECAST_ARTIFACTS_DIR, "long",  "long_forecast_lstm.csv"),
    "Bayesian VAR": os.path.join(FORECAST_ARTIFACTS_DIR, "long",  "long_forecast_bayesian_var.csv"),
}
ALL_INPUTS_SHORT = {
    "Prophet":      os.path.join(FORECAST_ARTIFACTS_DIR, "short", "short_forecast_prophet.csv"),
    "SARIMAX":      os.path.join(FORECAST_ARTIFACTS_DIR, "short", "short_forecast_sarimax.csv"),
    "LSTM":         os.path.join(FORECAST_ARTIFACTS_DIR, "short", "short_forecast_lstm.csv"),
    "Bayesian VAR": os.path.join(FORECAST_ARTIFACTS_DIR, "short", "short_forecast_bayesian_var.csv"),
}

ALL_OUT_LONG  = os.path.join(JSON_DIR, "forecast_all_long_line.json")
ALL_OUT_SHORT = os.path.join(JSON_DIR, "forecast_all_short_line.json")


# ---------------- Helpers ----------------
def _normalize_model_key(period: str, model: str) -> str:
    """
    Accept a full key (long_prophet/short_lstm) or a label (Prophet/LSTM/SARIMAX/BVAR),
    and normalize to 'long_<name>' or 'short_<name>' based on period.
    """
    m = (model or "").strip().lower()
    if m in ("all", "all_models"):
        return "ALL"
    if m.startswith(("long_", "short_")):
        return m

    label_map = {
        "prophet": "prophet",
        "sarimax": "sarimax",
        "lstm": "lstm",
        "bvar": "bayesian_var",
        "bayesian_var": "bayesian_var",
    }
    if m not in label_map:
        raise FileNotFoundError(f"Unknown model label: {model}")

    horizon = "short" if (period or "").upper() in SHORT_PERIODS else "long"
    return f"{horizon}_{label_map[m]}"

def _csv_for_model(model_key: str) -> str:
    """Map normalized key -> CSV path inside forecasts/<horizon>/"""
    mk = model_key.lower()
    horizon, name = mk.split("_", 1)
    if name == "bvar":
        name = "bayesian_var"
    fname = f"{horizon}_forecast_{name}.csv"
    # prefer artifacts/forecasts (generated outputs), fall back to web/forecasts
    candidate = os.path.join(FORECAST_ARTIFACTS_DIR, horizon, fname)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(FORECAST_ARTIFACTS_DIR, horizon, fname)

def _json_out(model_key: str) -> str:
    return os.path.join(JSON_DIR, f"forecast_{model_key}_candlestick.json")

def _no_cache_response(file_path: str):
    resp = make_response(send_file(file_path, mimetype="application/json"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


# ---------------- Routes ----------------
@app.route("/")
def home():
    tpl = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(tpl):
        return render_template("index.html")
    root = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(root):
        return send_from_directory(BASE_DIR, "index.html")
    return "<h3>App is running. Try /chart/1M/long_prophet.json</h3>"

# ---------------- Historical data ----------------
def _download_block(symbols, period, interval, auto_adjust=False):
    frames = []
    for sym in [s.strip() for s in symbols if s.strip()]:
        df_sym = yf.download(sym, period=period, interval=interval,
                             auto_adjust=auto_adjust, progress=False)
        if df_sym.empty:
            print(f"[PULL] WARN: no data for {sym}")
            continue
        df_sym = df_sym[["Open","High","Low","Close","Volume"]].copy()
        df_sym.columns = [f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close", f"{sym}_volume"]
        frames.append(df_sym)
    if not frames:
        raise RuntimeError("No data downloaded")
    df = pd.concat(frames, axis=1).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def _pull_short(symbols):
    """30d @ 1h; keep rows <=60% missing; interpolate + ffill + bfill; fill rest 0"""
    print("[PULL] short: 30d@1h")
    df_raw = _download_block(symbols, period="30d", interval="1h", auto_adjust=False)
    df = df_raw.copy()
    missing_pct = df.isna().sum(axis=1) / len(df.columns) * 100.0
    df = df[missing_pct <= 60]
    if df.empty:
        print("[PULL] WARN: filtered to empty; using raw for fill")
        df = df_raw.copy()
    df = df.interpolate("linear", limit_direction="both").ffill().bfill().fillna(0)
    out = os.path.join(STATIC_CHART, "short_ohlc.csv")
    df.index.name = "Date"
    df.to_csv(out)
    print(f"[PULL] wrote {out} rows={len(df)}")

def _pull_long(symbols):
    """2y @ 1d; drop rows with >20 NaNs; interpolate + ffill + bfill"""
    print("[PULL] long: 2y@1d")
    df = _download_block(symbols, period="2y", interval="1d", auto_adjust=False)
    df = df[df.isna().sum(axis=1) <= 20]
    df = df.interpolate("linear", limit_direction="both").ffill().bfill()
    out = os.path.join(STATIC_CHART, "long_ohlc.csv")
    df.index.name = "Date"
    df.to_csv(out)
    print(f"[PULL] wrote {out} rows={len(df)}")

def pull_csvs(symbols=None):
    """Run both pulls safely."""
    try:
        syms = symbols or DEFAULT_SYMBOLS
        _pull_short(syms)
        _pull_long(syms)
        print("[PULL] done.")
    except Exception as e:
        print(f"[PULL] ERROR: {e}")

def _fetch_yf(symbol: str, period: str, interval: str):
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return []
    df = df[["Open","High","Low","Close"]].copy()
    df.columns = ["open","high","low","close"]
    df = df.apply(pd.to_numeric, errors="coerce")
    # lenient cleaning for hourly
    missing_pct = df.isna().sum(axis=1) / len(df.columns) * 100.0
    df = df[missing_pct <= 60].copy() if not df.empty else df
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()
    idx = df.index
    try:
        idx = idx.tz_localize("UTC") if getattr(idx,"tz",None) is None else idx.tz_convert("UTC")
    except Exception:
        idx = pd.to_datetime(idx, utc=True, errors="coerce")
    df.insert(0, "time", idx.strftime("%Y-%m-%dT%H:%M:%SZ"))
    recs = df.reset_index(drop=True)[["time","open","high","low","close"]].to_dict("records")
    # coerce to plain floats
    for r in recs:
        r["open"]   = float(r["open"])   if r["open"]   is not None else None
        r["high"]   = float(r["high"])   if r["high"]   is not None else None
        r["low"]    = float(r["low"])    if r["low"]    is not None else None
        r["close"]  = float(r["close"])  if r["close"]  is not None else None
    return recs

def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[OK] wrote {path} ({len(payload.get('data', []))} rows)")

def build_history_json(symbol: str = "GC=F", outdir: str | None = None):
    base = Path(outdir) if outdir else Path(getattr(app, "static_folder", ".")) / "chart"
    short = _fetch_yf(symbol, "729d", "1h")
    for p in ("1H","24H","1W"): _write_json(base/p/"History.json", {"data": short})
    longd = _fetch_yf(symbol, "3y", "1d")
    for p in ("1M","3M","6M"): _write_json(base/p/"History.json", {"data": longd})
    return len(short), len(longd)

@app.before_request
def _prime_history_once():
    # run once per process
    if not app.config.get("_HISTORY_PRIMED", False):
        with _primed_lock:
            if not app.config.get("_HISTORY_PRIMED", False):
                try:
                    rows_s, rows_l = build_history_json(symbol="GC=F")
                    print(f"[PRIME] History JSON ready: hourly={rows_s}, daily={rows_l}")
                finally:
                    app.config["_HISTORY_PRIMED"] = True
# =========================================================

@app.route("/admin/rebuild-history")
def admin_rebuild_history():
    symbol = request.args.get("symbol", "GC=F")
    outdir = request.args.get("outdir")  # e.g. "static/chart"
    rows_s, rows_l = build_history_json(symbol=symbol, outdir=outdir)
    return jsonify({"ok": True, "symbol": symbol,
                    "written": {"1H": rows_s, "24H": rows_s, "1W": rows_s,
                                "1M": rows_l, "3M": rows_l, "6M": rows_l}})

def _load_json_file(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _rows_from_spec(spec: dict) -> list:
    """Extract rows from a forecast spec (support data.values or data.url)."""
    # 1) top-level values
    d = spec.get("data")
    if isinstance(d, dict) and isinstance(d.get("values"), list):
        return d["values"]

    # 2) first layer values
    for lyr in spec.get("layer", []):
        d = lyr.get("data")
        if isinstance(d, dict) and isinstance(d.get("values"), list):
            return d["values"]

    # 3) url (resolve relative to static/chart/)
    def _url_of(node):
        dd = node.get("data")
        return dd.get("url") if isinstance(dd, dict) and isinstance(dd.get("url"), str) else None

    url = _url_of(spec)
    if not url and isinstance(spec.get("layer"), list):
        for lyr in spec["layer"]:
            url = _url_of(lyr)
            if url:
                break

    if not url:
        return []

    # resolve local path
    if url.startswith("/static/"):
        local = Path(BASE_DIR) / url.lstrip("/")
    else:
        local = Path(JSON_DIR) / url  # common case: file sits in static/chart/
    if not local.exists():
        # last resort: try under static/ root
        alt = Path(BASE_DIR) / "static" / url
        if alt.exists():
            local = alt

    try:
        payload = _load_json_file(local)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
    except Exception:
        pass
    return []

def _map_ohlc_row(r: dict) -> dict:
    """Map {Date/Open/High/Low/Close} or {time/open/high/low/close} -> unified keys."""
    date = r.get("Date") or r.get("time")
    if date and "T" not in str(date):
        date = f"{date}T00:00:00"
    return {
        "Date":  date,
        "Open":  r.get("Open", r.get("open")),
        "High":  r.get("High", r.get("high")),
        "Low":   r.get("Low",  r.get("low")),
        "Close": r.get("Close",r.get("close")),
    }

def _parse_dt(s: str):
    try:
        return datetime.fromisoformat(str(s).replace("Z",""))
    except Exception:
        return None

def build_merged_json(period: str, model_label: str) -> str:
    """
    Build: static/chart/<period>/<model_label>_merged.json
    Combine history (<= earliest forecast date) + forecast rows.
    """
    per = (period or "").upper()
    forecast_spec_path = Path(JSON_DIR) / per / f"{model_label}.json"

    if not forecast_spec_path.exists():
        raise FileNotFoundError(f"Forecast spec not found: {forecast_spec_path}")

    # ---- load forecast rows ----
    spec = _load_json_file(forecast_spec_path)
    f_rows_raw = _rows_from_spec(spec)
    f_rows = [_map_ohlc_row(r) for r in f_rows_raw if isinstance(r, dict)]
    if not f_rows:
        raise RuntimeError("No forecast rows found in spec")

    # earliest forecast date
    earliest = None
    for r in f_rows:
        dt = _parse_dt(r.get("Date"))
        if dt and (earliest is None or dt < earliest):
            earliest = dt

    # ---- load history rows for this period ----
    hist_path = Path(JSON_DIR) / per / "History.json"
    if not hist_path.exists():
        raise FileNotFoundError(f"History not found for {per}: {hist_path}")
    hist_payload = _load_json_file(hist_path)
    h_rows_raw = hist_payload.get("data") if isinstance(hist_payload, dict) else hist_payload
    h_rows = [_map_ohlc_row(r) for r in (h_rows_raw or []) if isinstance(r, dict)]

    # keep history <= earliest
    if earliest:
        h_rows = [r for r in h_rows if (dt := _parse_dt(r.get("Date"))) and dt <= earliest]

    # tag + merge + sort
    for r in h_rows: r["source"] = "history"
    for r in f_rows: r["source"] = "forecast"
    merged = h_rows + f_rows
    merged.sort(key=lambda r: _parse_dt(r.get("Date")) or datetime.min)

    # write out
    out_dir = Path(JSON_DIR) / per
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_label}_merged.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "period": per,
            "model": model_label,
            "data": merged
        }, f, ensure_ascii=False)
    print(f"[MERGED] -> {out_path} ({len(merged)} rows)")
    return str(out_path)

@app.route("/chart/<period>/<model>.json")
def chart_by_period_model(period, model):
    """
    Build on demand every time the URL is hit:
      - model == 'All'  -> build_all_models_line(...) to static/chart/forecast_all_*.json
      - otherwise       -> generate_forecast_chart(...) candlestick for that model
    Returns the JSON with no-cache headers.
    """
    model_norm = _normalize_model_key(period, model)

    # ---- ALL models line chart ----
    # if model_norm == "ALL":
    #     is_short = (period or "").upper() in SHORT_PERIODS
    #     inputs_map = ALL_INPUTS_SHORT if is_short else ALL_INPUTS_LONG
    #     out_path   = ALL_OUT_SHORT  if is_short else ALL_OUT_LONG

    #     print(f"[REBUILD][ALL] period={period} -> {out_path}")
    #     try:
    #         # build_all_models_line(period: 'short'|'long', inputs_map, output_json_path)
    #         build_all_models_line("short" if is_short else "long", inputs_map, out_path)
    #     except Exception as e:
    #         traceback.print_exc()
    #         abort(500, description=f"Failed to build ALL chart: {e}")

    #     if not os.path.exists(out_path):
    #         abort(500, description=f"Generated ALL JSON missing: {out_path}")
    #     return _no_cache_response(out_path)

    # ---- Single model candlestick ----
    csv_path = _csv_for_model(model_norm)
    if not os.path.exists(csv_path):
        abort(404, description=f"CSV not found for {model_norm}: {csv_path}")

    print(f"[REBUILD] {model_norm} <- {csv_path}")
    try:
        generate_forecast_chart(model_norm, csv_path, output_dir="static/chart")
    except Exception as e:
        traceback.print_exc()
        abort(500, description=f"Failed to build chart for {model_norm}: {e}")

    out_path = _json_out(model_norm)
    if not os.path.exists(out_path):
        abort(500, description=f"Generated JSON missing: {out_path}")
    return _no_cache_response(out_path)

@app.route("/admin/build-merged")
def admin_build_merged():
    period = request.args.get("period", "1W")
    model  = request.args.get("model", "BVAR")   # labels: Prophet | SARIMAX | LSTM | BVAR
    try:
        out_path = build_merged_json(period, model)
        return jsonify({"ok": True, "period": period, "model": model, "merged": out_path})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------------- Small APIs (unchanged) ----------------
def fetch_data_with_retry(ticker: str, retries: int = 3, delay_sec: int = 5):
    for i in range(retries):
        try:
            df = yf.download(ticker, period='7d', interval='1m')
            if not df.empty:
                return df
            print(f"[{i+1}/{retries}] empty data for {ticker}")
        except Exception as e:
            print(f"[{i+1}/{retries}] error: {e}")
        time.sleep(delay_sec)
    return None

@app.route('/api/gold-price')
def get_gold_price():
    try:
        ticker = 'GC=F'
        data = fetch_data_with_retry(ticker)
        if data is None or data.empty:
            return jsonify({"error": "No data"}), 500

        latest_row = data.iloc[-1]
        ts = data.index[-1]
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        return jsonify({
            "ticker": ticker,
            "latest_price": float(latest_row['Close']),
            "timestamp": ts_str
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/market-trends')
def get_market_trends():
    return jsonify({'1H': 58, '24H': 41, '1W': 73, '1M': 73})

# Avoid double-run under the dev reloader
def _should_bootstrap():
    # when using `flask run`, WERKZEUG_RUN_MAIN == 'true' only in the reloaded process
    val = os.environ.get("WERKZEUG_RUN_MAIN")
    return (val == "true") or (val is None)

@app.before_request
def _auto_pull_once():
    # run exactly once (per process) on the first incoming request
    if not _bootstrap_started.is_set():
        # avoid double-run under the dev reloader; still harmless if it happens
        main = os.environ.get("WERKZEUG_RUN_MAIN")
        if (main == "true") or (main is None):
            _bootstrap_started.set()
            threading.Thread(
                target=pull_csvs,              # your function that pulls CSVs
                kwargs={"symbols": DEFAULT_SYMBOLS},
                daemon=True
            ).start()

@app.route("/admin/rebuild-csv")
def admin_rebuild_csv():
    syms = request.args.get("symbols")
    symbols = syms.split(",") if syms else DEFAULT_SYMBOLS
    threading.Thread(target=pull_csvs, kwargs={"symbols": symbols}, daemon=True).start()
    return jsonify({"ok": True, "status": "pull started", "symbols": symbols})

# ---------------- Run ----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", "5000")))