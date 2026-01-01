document.addEventListener('DOMContentLoaded', () => {
  const dateDay = document.querySelector('.date-box .day');
  const dateMonth = document.querySelector('.date-box .month');
  const dateTime = document.querySelector('.date-box .time');
  let currentView = null;

  function updateLocalTime(){
    const now = new Date();
    dateDay.textContent = now.getDate();
    dateMonth.textContent = now.toLocaleString('default',{month:'short'});
    dateTime.textContent = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;
  }

  function fetchGoldPrice(){
    fetch('/api/gold-price')
      .then(r=>r.json())
      .then(data=>{
        if(data.latest_price){
          document.getElementById('current-price').innerText = `$${(+data.latest_price).toFixed(2)}`;
          if(data.timestamp){
            const thai = new Intl.DateTimeFormat('th-TH',{year:'numeric',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false,timeZone:'Asia/Bangkok'}).format(new Date(data.timestamp));
            document.getElementById('price-timestamp').innerText = `Last Update: ${thai}`;
          }
        }else{
          document.getElementById('current-price').innerText = data.message || 'Failed to fetch price';
        }
      })
      .catch(()=>{ document.getElementById('current-price').innerText = 'Failed to fetch price'; });
  }

  function fetchMarketTrends() {
      fetch('/api/market-trends')
          .then(res => res.json())
          .then(data => {
              const list = document.getElementById('market-trends-list');
              list.innerHTML = ''; // Clear old trends
              for (const [period, value] of Object.entries(data)) {
                  const item = document.createElement('li');
                  const progressClass = value >= 50 ? 'positive' : 'negative';
                  item.innerHTML = `
                      <span>${period}</span>
                      <div class="progress-bar-bg">
                          <div class="progress-bar-fill ${progressClass}" style="width:${value}%;"></div>
                      </div>
                      <span>${value}%</span>
                  `;
                  list.appendChild(item);
              }
          })
          .catch(err => console.error('Failed to fetch market trends:', err));
  }
  
  function showLivePlaceholder(){
    const el = document.getElementById('vega-chart-container');
    el.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:center;height:100%;color:#E8C766;font-weight:700;">
        Live chart will be connected to API later.
      </div>`;
  }

  function showAllModelsPlaceholder(){
    const el = document.getElementById('vega-chart-container');
    el.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:center;height:100%;color:#E8C766;font-weight:700;text-align:center;">
        Please select a specific model<br>to view the forecast chart
      </div>`;
  }

  function ensureIndicatorDom(){
    let wrap = document.getElementById('indicator');
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.id = 'indicator';
      wrap.style.cssText = 'margin-top:8px;color:#e6edf6;font-size:16px;display:flex;gap:18px;';
      document.getElementById('vega-chart-container')?.after(wrap);
    }
    if (!document.querySelector('#indicator-high')) {
      const hi = document.createElement('div');
      hi.id = 'indicator-high';
      hi.innerHTML = 'Highest: <strong class="value">-</strong>';
      wrap.appendChild(hi);
    }
    if (!document.querySelector('#indicator-low')) {
      const lo = document.createElement('div');
      lo.id = 'indicator-low';
      lo.innerHTML = 'Lowest: <strong class="value">-</strong>';
      wrap.appendChild(lo);
    }
  }

  function setupClickIndicator(res){
    // 1) DOM
    ensureIndicatorDom();
    const elHigh = document.querySelector('#indicator-high .value');
    const elLow  = document.querySelector('#indicator-low  .value');
    const fmt = n => Number(n).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});

    // 2) ดึงข้อมูลจาก Vega view
    let rows = [];
    try {
      const names = (typeof res.view.dataNames === 'function') ? res.view.dataNames() : [];
      const preferred = ['source_0','data_0', ...names];
      const seen = new Set();
      for (const nm of preferred) {
        if (!nm || seen.has(nm)) continue;
        seen.add(nm);
        const arr = res.view.data(nm) || [];
        if (Array.isArray(arr) && arr.length && 'Date' in arr[0] && 'High' in arr[0] && 'Low' in arr[0]) {
          rows = arr.map(d=>({...d}));
          break;
        }
      }
    } catch(e){}

    if (!rows.length && res.spec && res.spec.data && Array.isArray(res.spec.data.values)) {
      const arr = res.spec.data.values;
      if (arr?.length && 'Date' in arr[0] && 'High' in arr[0] && 'Low' in arr[0]) rows = arr.slice();
    }
    if (!rows.length) { elHigh.textContent='-'; elLow.textContent='-'; return; }

    // 3) เตรียม timestamp
    for (const d of rows) d.__t = new Date(d.Date).getTime();

    // 4) อ่านโดเมน X ปัจจุบัน (รองรับซูม/แพน)
    function currentXDomain(){
      let x0 = Math.min(...rows.map(r=>r.__t));
      let x1 = Math.max(...rows.map(r=>r.__t));
      try {
        const st = res.view.getState({signals:'raw'}) || {};
        const sigs = Object.keys(st.signals || {});
        const sig = sigs.find(n => /^zoom.*_x$/.test(n)) || null; // ชื่อสัญญาณซูมที่ Altair ตั้งไว้
        if (sig) {
          const v = res.view.signal(sig);
          if (Array.isArray(v)) [x0, x1] = v;
          else if (v && Array.isArray(v.extent)) [x0, x1] = v.extent;
        }
      } catch(e){}
      return [x0, x1];
    }

    const visibleRows = () => {
      const [x0,x1] = currentXDomain();
      return rows.filter(r => r.__t >= x0 && r.__t <= x1);
    };

    function setMinMax(dataSubset){
      if (!dataSubset.length) { elHigh.textContent='-'; elLow.textContent='-'; return; }
      let hi = -Infinity, lo = Infinity;
      for (const r of dataSubset) {
        const H = +r.High, L = +r.Low;
        if (Number.isFinite(H) && H > hi) hi = H;
        if (Number.isFinite(L) && L < lo) lo = L;
      }
      elHigh.textContent = Number.isFinite(hi) ? fmt(hi) : '-';
      elLow.textContent  = Number.isFinite(lo) ? fmt(lo) : '-';
    }

    // 5) ค่าเริ่มต้น & ดับเบิลคลิก = min/max ของช่วงที่มองเห็น
    function resetIndicator(){ setMinMax(visibleRows()); }
    resetIndicator();

    // 6) หาแท่งใกล้ตำแหน่งคลิก (รองรับทั้ง canvas และ svg)
    function getChartElement(){
      const container = document.getElementById('vega-chart-container');
      return container?.querySelector('canvas') || container?.querySelector('svg') || null;
    }
    function nearestRowFromClick(ev){
      const el = getChartElement();
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      const xPx  = ev.clientX - rect.left;
      const [x0,x1] = currentXDomain();
      const t = x0 + (x1 - x0) * (xPx / rect.width);
      let best=null, bestAbs=Infinity;
      for (const d of rows) {
        if (d.__t < x0 || d.__t > x1) continue;
        const a = Math.abs(d.__t - t);
        if (a < bestAbs) { bestAbs = a; best = d; }
      }
      return best;
    }

    // 7) bind events (ป้องกันซ้ำ)
    const chartEl = getChartElement();
    if (chartEl) {
      chartEl.onclick = null;
      chartEl.ondblclick = null;

      chartEl.addEventListener('click', ev => {
        const row = nearestRowFromClick(ev);
        if (row) {
          elHigh.textContent = fmt(+row.High);
          elLow.textContent  = fmt(+row.Low);
        }
      });

      chartEl.addEventListener('dblclick', () => {
        resetIndicator();
      });
    }

    // 8) อัปเดตเมื่อซูม/แพน
    try {
      const st = res.view.getState({signals:'raw'}) || {};
      const sigs = Object.keys(st.signals || {});
      const sig = sigs.find(n => /^zoom.*_x$/.test(n));
      if (sig) {
        res.view.removeSignalListener(sig, window.__indicatorZoomListener);
        window.__indicatorZoomListener = () => resetIndicator();
        res.view.addSignalListener(sig, window.__indicatorZoomListener);
      }
    } catch(e){}
  }

  window.currentPeriod = 'LIVE';
  window.currentModel  = (document.getElementById('model-select')?.value) || 'All';
  renderLiveTV({ symbol: 'TVC:GOLD', interval: '60' });

  function renderForecastChart(period, model){
    // Clear container before embedding new chart
    document.getElementById('vega-chart-container').innerHTML = '';

    // If-else logic for buttons and dropdown
    if(period === 'LIVE'){
      showLivePlaceholder();
      return;
    } else if(model === 'All'){
      renderAllChart();
      return;
    } else {
      // Valid period and model selected - load the chart
      const m = model || window.currentModel;          
      const url = `/chart/${encodeURIComponent(period)}/${encodeURIComponent(m)}.json`;
      const opt = {
        actions:{ export:true, source:false, compiled:false, editor:false },
        theme:'dark', renderer:'canvas', hover:true,
        tooltip:{ theme:'dark', format:'html' }
      };

      vegaEmbed('#vega-chart-container', url, opt)
        .then(res => { 
          currentView = res.view;
          setupClickIndicator(res);  
          setTimeout(()=>res.view.resize().run(), 80);
        })
        .catch(err=>{
          console.error(err);
          document.getElementById('vega-chart-container').innerHTML =
            `<div style="display:flex;align-items:center;justify-content:center;height:100%;flex-direction:column;">
               <p style="color:var(--negative-red);font-size:1.2rem;margin-bottom:10px;">Error loading chart for ${period} - ${m}</p>
               <p style="color:#99A1C4;font-size:.9rem;">URL: ${url}</p>
             </div>`;
        });
    }
  }

  function sanitizeSpecForAll(spec){
    function walk(node){
      if (!node || typeof node!=='object') return;

      // (1) ลบ transform/filter ที่อาจซ่อน Historical
      if (Array.isArray(node.transform)) {
        node.transform = node.transform.filter(t => {
          const s = String(t?.filter ?? t?.expr ?? '');
          if (/Model\s*(!==|!=)\s*['"]Historical['"]/.test(s)) return false;
          if (t && typeof t.filter === 'object' && t.filter.field === 'Model' && Array.isArray(t.filter.oneOf)) {
            const hasHist = t.filter.oneOf.includes('Historical');
            if (!hasHist) return false;
          }
          return true;
        });
      }

      // (2) อย่าบังคับ x-scale domain (กันโดนบีบจนไม่เห็นเส้นยาวของ Historical)
      if (node.encoding && node.encoding.x && node.encoding.x.scale && node.encoding.x.scale.domain){
        delete node.encoding.x.scale.domain;
      }

      // เดินทุก child
      ['layer','hconcat','vconcat','facet','spec','repeat'].forEach(k=>{
        const v = node[k];
        if (Array.isArray(v)) v.forEach(walk);
        else if (v && typeof v==='object') walk(v);
      });
    }
    walk(spec);
    return spec;
  }

  async function renderAllChart() {
    if (typeof showTV === 'function') showTV(false);

    const url = currentJsonForAll() + '?t=' + Date.now(); // cache-buster
    try {
      const res  = await fetch(url, {cache:'no-store'});
      if (!res.ok) throw new Error('Spec fetch failed: ' + res.status);
      const spec = await res.json();

      // สำคัญ: ล้างฟิลเตอร์/โดเมนที่ซ่อน Historic
      const clean = sanitizeSpecForAll(spec);

      // (กัน overlay/สไตล์) — เคลียร์และให้คอนเทนเนอร์มองเห็นได้แน่นอน
      const el = document.querySelector('#vega-chart-container');
      el.style.minHeight = '380px';
      el.innerHTML = '';

      await vegaEmbed('#vega-chart-container', clean, {
        actions:{export:true, source:false, compiled:false, editor:false},
        renderer:'canvas', theme:'dark', tooltip:{theme:'dark'}
      });
    } catch (e) {
      console.error('ALL chart load failed:', e);
      document.querySelector('#vega-chart-container').innerHTML =
        `<div style="color:#ff6b6b;padding:12px;">โหลด ALL chart ไม่สำเร็จ<br>${url}</div>`;
    }
  }

  function showTV(on){
    const tv   = document.getElementById('tv-chart-container');
    const vega = document.getElementById('vega-chart-container');
    if (!tv || !vega) return;
    tv.style.display   = on ? 'block' : 'none';
    vega.style.display = on ? 'none'  : 'block';
    setTimeout(()=>window.dispatchEvent(new Event('resize')),0);
  }

  let __tvScriptLoading = null;
  function loadTVScriptOnce(){
    if (window.TradingView) return Promise.resolve();
    if (__tvScriptLoading)  return __tvScriptLoading;
    __tvScriptLoading = new Promise((resolve,reject)=>{
      const s=document.createElement('script');
      s.src='https://s3.tradingview.com/tv.js';
      s.async=true;
      s.onload=resolve; s.onerror=()=>reject(new Error('Failed to load tv.js'));
      document.head.appendChild(s);
    });
    return __tvScriptLoading;
  }

let __tvWidget = null;
let __tvWidgetCreated = false;

async function renderLiveTV({
    containerId = 'tv-chart-container',
    symbol = 'TVC:GOLD',
    interval = '60',
    timezone = 'Asia/Bangkok',
    theme = 'dark',
    locale = 'en'
} = {}) {
    showTV(true);
    await loadTVScriptOnce();
    
    const el = document.getElementById(containerId);
    if (!el) return;

    // Clear previous widget if exists
    if (__tvWidget) {
        __tvWidget.remove();
        __tvWidget = null;
    }

    el.innerHTML = '';

    /* global TradingView */
    __tvWidget = new TradingView.widget({
        autosize: true,
        symbol: symbol,
        interval: interval,
        timezone: timezone,
        theme: theme,
        style: "1",
        locale: locale,
        toolbar_bg: "#0F2043",
        enable_publishing: false,
        allow_symbol_change: false,
        container_id: containerId,
        height: "100%",
        width: "100%",
        hide_top_toolbar: false,
        hide_side_toolbar: true,
        withdateranges: true,
        studies: [
            "BB@tv-basicstudies",
            "RSI@tv-basicstudies"
        ],
        show_popup_button: true,
        popup_width: "1000",
        popup_height: "650",
        details: false,
        hotlist: false,
        calendar: false,
        news: [
            "headlines"
        ]
    });

    __tvWidgetCreated = true;

    // Force resize after a short delay to ensure proper rendering
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 100);

    // Set up resize observer for the chart container
    const holder = document.querySelector('.chart-placeholder');
    if (holder && 'ResizeObserver' in window) {
        const resizeObserver = new ResizeObserver(() => {
            window.dispatchEvent(new Event('resize'));
            // Force TradingView to recalculate its dimensions
            if (__tvWidget && typeof __tvWidget.onResize === 'function') {
                setTimeout(() => {
                    __tvWidget.onResize();
                }, 50);
            }
        });
        resizeObserver.observe(holder);
    }
}

  function leaveLive(period, model){
    showTV(false);
    // วาดกราฟเดิมของคุณ
    if (typeof renderForecastChart === 'function'){
      renderForecastChart(period, model);
    }
  }

  async function renderLiveChart({
    ticker = 'GC=F', 
    interval = '1m',
    period = '1d'
    } = {}) {
    const url = `/api/live-candles?ticker=${encodeURIComponent(ticker)}&interval=${encodeURIComponent(interval)}&period=${encodeURIComponent(period)}`;

    try {
        const res = await fetch(url);
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || 'fetch failed');
        const values = payload.data || [];

        // สเปค candlestick (Vega-Lite)
        const spec = {
        $schema: "https://vega.github.io/schema/vega-lite/v5.json",
        width: "container",
        height: "container",
        background: "transparent",
        data: { values },
        encoding: {
            x: { field: "time", type: "temporal", title: null, axis:{format: "%H:%M"} }
        },
        layer: [
            { // ไส้เทียน
            mark: { type: "rule" },
            encoding: {
                y: { field: "low",  type: "quantitative", title: "Price (USD)" },
                y2:{ field: "high" },
                color: { value: "#7dd3fc" }
            }
            },
            { // แท่ง open-close
            mark: { type: "bar" },
            encoding: {
                y:  { field: "open",  type: "quantitative" },
                y2: { field: "close" },
                color: {
                condition: { test: "datum.close >= datum.open", value: "#4cc38a" }, // เขียวขึ้น
                value: "#e54d2e" // แดงลง
                },
                tooltip: [
                {field:"time", type:"temporal", title:"Time"},
                {field:"open",  type:"quantitative", title:"Open"},
                {field:"high",  type:"quantitative", title:"High"},
                {field:"low",   type:"quantitative", title:"Low"},
                {field:"close", type:"quantitative", title:"Close"},
                {field:"volume", type:"quantitative", title:"Vol"}
                ]
            }
            }
        ]
        };

        await vegaEmbed('#vega-chart-container', spec, {
          actions:{export:true, source:false, compiled:false, editor:false},
          renderer:'canvas', theme:'dark', hover:true,
          tooltip:{ theme:'dark', format:'html' }
        }).then(r => {
          setupClickIndicator({ view: r.view, spec });
          setTimeout(() => r.view.resize().run(), 60);
        });
        
    } catch(err) {
        console.error(err);
        document.getElementById('vega-chart-container').innerHTML =
        `<div style="display:flex;align-items:center;justify-content:center;height:100%;flex-direction:column;">
            <p style="color:var(--negative-red);font-size:1.2rem;margin-bottom:10px;">Live chart error</p>
            <p style="color:#99A1C4;font-size:.9rem;">${url}</p>
        </div>`;
    }
}

  // --- map period -> time window ---
  function getRangeForPeriod(period){
    var now = new Date();
    var ms =
      period === '1H' ?   1*60*60*1000 :
      period === '24H'?  24*60*60*1000 :
      period === '1W' ?   7*24*60*60*1000 :
      period === '1M' ?  30*24*60*60*1000 :
      period === '3M' ?  90*24*60*60*1000 :
                          180*24*60*60*1000; // default 6M
    var startDt = new Date(now.getTime() - ms);
    var endDt   = now;
    function ymd(d){ return d.toISOString().slice(0,10); }
    var isLong = (period === '1M' || period === '3M' || period === '6M');
    return isLong
      ? { start: ymd(startDt), end: ymd(endDt) }               // date-only
      : { start: startDt.toISOString(), end: endDt.toISOString() }; // datetime
  }

  function currentJsonForAll(){
    switch (window.currentPeriod) {
      case '1H':  return '/static/chart/forecast_all_1H.json';
      case '24H': return '/static/chart/forecast_all_24H.json';
      case '1W':  return '/static/chart/forecast_all_1W.json';
      case '1M':  return '/static/chart/forecast_all_1M.json';
      case '3M':  return '/static/chart/forecast_all_3M.json';
      case '6M':  return '/static/chart/forecast_all_6M.json';
      default:    return '/static/chart/forecast_all_1W.json';
    }
  }

  // --- fetch spec, cut [start,end], fix x-domain, render ---
  async function renderSpecWithRange(url, startISO, endISO){
      if (typeof showTV === 'function') showTV(false);

      const res = await fetch(url + (url.indexOf('?')>-1 ? '&' : '?') + 't=' + Date.now(), { cache:'no-store' });
      if (!res.ok) throw new Error('Spec fetch failed: ' + res.status);
      const spec = await res.json();

      // Debug: Log original data count
      function* rowsOf(node){
          if (!node || typeof node !== 'object') return;
          if (node.data && Array.isArray(node.data.values)) yield* node.data.values;
          const keys = ['layer','hconcat','vconcat','facet','spec','repeat'];
          for (let k of keys){
              if (Array.isArray(node[k])) for (let n of node[k]) yield* rowsOf(n);
              else if (node[k] && typeof node[k] === 'object') yield* rowsOf(node[k]);
          }
      }
      
      const all = [];
      for (const r of rowsOf(spec)) all.push(r);
      if (spec.datasets) {
          for (const k in spec.datasets) if (Array.isArray(spec.datasets[k])) all.push(...spec.datasets[k]);
      }
      
      console.log(`[DEBUG] Total rows before filtering: ${all.length}`);
      console.log(`[DEBUG] Period: ${window.currentPeriod}, URL: ${url}`);
      
      const toJS = (v)=>{ 
          if(v==null) return null; 
          const s=String(v); 
          return new Date(s.indexOf('T')>=0 ? s : s+'T00:00:00Z'); 
      };
      
      let dataMax = null;
      for (const r of all){
          const v = r && (r.Date != null ? r.Date : r.time);
          const d = toJS(v);
          if (d && (!dataMax || d > dataMax)) dataMax = d;
      }
      
      console.log(`[DEBUG] Max date in data: ${dataMax}`);
      console.log(`[DEBUG] Requested range: ${startISO} to ${endISO}`);

      // FIX: Use the provided range for short periods, don't recalculate
      const p = String(window.currentPeriod || '').trim();
      const isLong = (p === '1M' || p === '3M' || p === '6M');
      
      let startB, endB, xStart, xEnd;
      
      if (isLong && dataMax){
          // Only use the long-period logic for actual long periods
          const dayMs = 24*60*60*1000;
          const lookback = (p==='1M'?30:p==='3M'?90:180)*dayMs;
          const end = new Date(Date.UTC(dataMax.getUTCFullYear(), dataMax.getUTCMonth(), dataMax.getUTCDate()+1, 0,0,0));
          const start = new Date(end.getTime() - lookback);
          startB = start; 
          endB = end;
          xStart = start.toISOString().slice(0,10);
          xEnd = new Date(end.getTime()-1).toISOString().slice(0,10);
      } else {
          // For short periods (1H, 24H, 1W), use the provided range directly
          startB = new Date(startISO);
          endB = new Date(endISO);
          xStart = startB.toISOString();
          xEnd = endB.toISOString();
      }
      
      console.log(`[DEBUG] Final range: ${xStart} to ${xEnd}`);
      console.log(`[DEBUG] isLong: ${isLong}`);

      // Fix the filtering logic
      function keepRow(r){
          if (!r) return false;
          const raw = (r.Date != null ? r.Date : r.time);
          if (raw == null) return true; // keep undated rows
          
          const d = toJS(raw);
          if (!d) return false;
          
          if (isLong){
              const day = new Date(d.toISOString().slice(0,10)+'T00:00:00Z');
              return (day >= startB && day < endB);
          }
          
          // For short periods, use direct date comparison
          return (d >= startB && d <= endB);
      }

      let keptRows = 0;
      const filterValues = (arr) => {
          if (!Array.isArray(arr)) return arr;
          const filtered = arr.filter(r => {
              const keep = keepRow(r);
              if (keep) keptRows++;
              return keep;
          });
          return filtered;
      };

      function walk(node){
          if (!node || typeof node !== 'object') return;
          
          if (node.data && Array.isArray(node.data.values)){
              const originalCount = node.data.values.length;
              node.data.values = filterValues(node.data.values);
              console.log(`[DEBUG] Filtered dataset: ${originalCount} -> ${node.data.values.length} rows`);
          }
          
          if (node.encoding && node.encoding.x){
              node.encoding.x.scale = Object.assign({}, node.encoding.x.scale || {}, { 
                  domain: [xStart, xEnd] 
              });
          }
          
          const keys = ['layer','hconcat','vconcat','facet','spec','repeat'];
          for (let k of keys){
              if (Array.isArray(node[k])) node[k].forEach(walk);
              else if (node[k] && typeof node[k] === 'object') walk(node[k]);
          }
      }
      
      walk(spec);
      
      if (spec.datasets){
          for (const k in spec.datasets){
              if (Array.isArray(spec.datasets[k])) {
                  const originalCount = spec.datasets[k].length;
                  spec.datasets[k] = filterValues(spec.datasets[k]);
                  console.log(`[DEBUG] Filtered dataset ${k}: ${originalCount} -> ${spec.datasets[k].length} rows`);
              }
          }
      }
      
      console.log(`[DEBUG] Total rows kept after filtering: ${keptRows}`);

    // dark theme fallback (keeps labels white)
    spec.config = spec.config || {};
    spec.config.axis = Object.assign({
      labelColor:'#ecf0f1', titleColor:'#ecf0f1',
      gridColor:'#34495e',  domainColor:'#7f8c8d', tickColor:'#7f8c8d'
    }, spec.config.axis || {});
    spec.config.legend = Object.assign({ labelColor:'#ecf0f1', titleColor:'#ecf0f1' }, spec.config.legend || {});
    spec.config.title  = Object.assign({ color:'#ecf0f1' }, spec.config.title  || {});

    await vegaEmbed('#vega-chart-container', spec, {
      actions:{ export:true, source:false, compiled:false, editor:false },
      renderer:'canvas',
      theme:'dark',
      tooltip:{ theme:'dark' }
    });
  }

  async function renderSpecCutFromEnd(url, periodStr){
    if (typeof showTV === 'function') showTV(false);

    const res = await fetch(url + (url.includes('?')?'&':'?') + 't=' + Date.now(), {cache:'no-store'});
    if (!res.ok) throw new Error('Spec fetch failed: '+res.status);
    const spec = await res.json();

    function* rowsOf(node){
      if (!node || typeof node !== 'object') return;
      if (node.data && Array.isArray(node.data.values)) yield* node.data.values;
      const ks = ['layer','hconcat','vconcat','facet','spec','repeat'];
      for (const k of ks){
        if (Array.isArray(node[k])) for (const n of node[k]) yield* rowsOf(n);
        else if (node[k] && typeof node[k]==='object') yield* rowsOf(node[k]);
      }
    }
    const all = [];
    for (const r of rowsOf(spec)) all.push(r);
    if (spec.datasets){
      for (const k in spec.datasets){
        if (Array.isArray(spec.datasets[k])) all.push(...spec.datasets[k]);
      }
    }
    if (!all.length) throw new Error('No data rows in spec');

    const toDate = v => {
      if (v==null) return null;
      const s = String(v);
      return new Date(s.includes('T') ? s : (s + 'T00:00:00Z'));
    };
    let dataMax = null;
    for (const r of all){
      const raw = r.Date ?? r.time;
      const d = toDate(raw);
      if (d && (!dataMax || d > dataMax)) dataMax = d;
    }
    if (!dataMax) throw new Error('Cannot detect max date');

    const p = String(periodStr||'').trim();
    const day = 24*60*60*1000, hour = 60*60*1000;
    const lookbackMs =
        p==='1H' ? 1*hour :
        p==='24H'? 24*hour :
        p==='1W' ? 7*day  :
        p==='1M' ? 30*day :
        p==='3M' ? 90*day :
                  180*day; // 6M
    const isLong = (p==='1M'||p==='3M'||p==='6M');

    let startB, endB, xStart, xEnd;
    if (isLong){
      const end = new Date(Date.UTC(
        dataMax.getUTCFullYear(), dataMax.getUTCMonth(), dataMax.getUTCDate()+1, 0,0,0
      ));
      const start = new Date(end.getTime() - lookbackMs);
      startB = start; endB = end;
      xStart = start.toISOString().slice(0,10);                 
      xEnd   = new Date(end.getTime()-1).toISOString().slice(0,10);
    }else{
      const end = dataMax;
      const start = new Date(end.getTime() - lookbackMs);
      startB = start; endB = end;
      xStart = start.toISOString();
      xEnd   = end.toISOString();
    }

    const keepRow = (r)=>{
      if (!r) return false;
      const raw = r.Date ?? r.time;
      if (raw == null) return true;
      const d = toDate(raw);
      if (!d) return false;
      if (isLong){
        const dayOnly = new Date(d.toISOString().slice(0,10)+'T00:00:00Z');
        return (dayOnly >= startB && dayOnly < (endB)); 
      }
      return (d >= startB && d <= endB);               
    };
    const filterValues = (arr)=> Array.isArray(arr) ? arr.filter(keepRow) : arr;

    function tuneNode(node){
      if (!node || typeof node!=='object') return;

      if (node.data && Array.isArray(node.data.values)){
        node.data.values = filterValues(node.data.values);
      }

      if (node.encoding && node.encoding.x){
        const xEnc = node.encoding.x;
        xEnc.scale = Object.assign({}, xEnc.scale||{}, { domain:[xStart, xEnd] });

        if (isLong){
          xEnc.timeUnit = 'yearmonthdate';
          xEnc.axis = Object.assign({}, xEnc.axis||{}, { format: '%Y-%m-%d', tickCount: 8 });
        }else{
          delete xEnc.timeUnit;
          xEnc.axis = Object.assign({}, xEnc.axis||{}, { format: '%Y-%m-%d %H:%M', tickCount: 8 });
        }
        node.encoding.x = xEnc;
      }

      if (node.encoding && Array.isArray(node.encoding.tooltip)){
        node.encoding.tooltip = node.encoding.tooltip.map(t=>{
          if (t && (t.field==='Date' || t.field==='time') && (t.type==='temporal' || t.type==='T' || t.type==='temporal')){
            const fmt = isLong ? '%Y-%m-%d' : '%Y-%m-%d %H:%M';
            return Object.assign({}, t, { format: fmt });
          }
          return t;
        });
      }

      if (node.mark){
        const mark = (typeof node.mark === 'string') ? {type: node.mark} : Object.assign({}, node.mark);
        if ((mark.type||'').toLowerCase()==='line'){
          if (isLong){
            mark.point = Object.assign({ filled: true, size: 30 }, (mark.point||{}));
          }else{
            if (mark.point) delete mark.point;
          }
        }
        node.mark = mark.type ? mark : node.mark;
      }

      const ks = ['layer','hconcat','vconcat','facet','spec','repeat'];
      for (const k of ks){
        if (Array.isArray(node[k])) node[k].forEach(tuneNode);
        else if (node[k] && typeof node[k]==='object') tuneNode(node[k]);
      }
    }
    tuneNode(spec);

    if (spec.datasets){
      for (const k in spec.datasets){
        if (Array.isArray(spec.datasets[k])) spec.datasets[k] = filterValues(spec.datasets[k]);
      }
    }

    spec.config = spec.config || {};
    spec.config.axis   = Object.assign({labelColor:'#ecf0f1', titleColor:'#ecf0f1'}, spec.config.axis||{});
    spec.config.legend = Object.assign({labelColor:'#ecf0f1', titleColor:'#ecf0f1'}, spec.config.legend||{});
    spec.config.title  = Object.assign({color:'#ecf0f1'}, spec.config.title||{});

    await vegaEmbed('#vega-chart-container', spec, {
      actions:{export:true, source:false, compiled:false, editor:false},
      renderer:'canvas', theme:'dark', tooltip:{theme:'dark'}
    });
  }

  // --- central: pick URL from currentPeriod/currentModel then cut ---
  async function renderCurrentWithButtonRange(){
    if (window.currentPeriod === 'LIVE') { renderLiveTV(); return; }

    if (window.currentModel === 'All') {
      // ❌ ห้ามเรียก renderSpecWithRange()/renderSpecCutFromEnd() ที่นี่
      await renderAllChart();
      return;
    }

    // โมเดลเดี่ยว → ใช้ endpoint เดิม + ตัดช่วงให้ตรงปุ่ม
    const rng = getRangeForPeriod(window.currentPeriod);
    const url = '/chart/' + encodeURIComponent(window.currentPeriod)
                        + '/' + encodeURIComponent(window.currentModel) + '.json';
    await renderSpecWithRange(url, rng.start, rng.end);
  }

  // period buttons
  var tagButtons = document.querySelectorAll('.tag-buttons .btn-period');
  tagButtons.forEach(function(btn){
    btn.addEventListener('click', async function(){
      window.currentPeriod = btn.dataset.period || btn.textContent.trim();   // ← ใช้ window.

      tagButtons.forEach(function(b){ b.classList.remove('active'); });
      btn.classList.add('active');

      if (window.currentPeriod === 'LIVE') {                                  // ← ใช้ window.
        renderLiveTV();
        return;
      }
      try {
        await renderCurrentWithButtonRange();
      } catch (e) {
        console.error(e);
        document.getElementById('vega-chart-container').innerHTML =
          '<div style="color:#ff6b6b;padding:12px;">Failed to render range</div>';
      }
    });
  });

  // model dropdown
  const modelSel = document.getElementById('model-select');
  window.currentModel = modelSel.value || 'All';                               // ← ใช้ window.

  modelSel.addEventListener('change', async function(e){
    window.currentModel = e.target.value;                                      // ← ใช้ window.
    if (window.currentPeriod === 'LIVE') {
      renderLiveTV();
    } else {
      try { await renderCurrentWithButtonRange(); } catch(e){ console.error(e); }
    }
  });

  window.currentPeriod = 'LIVE';
  renderLiveTV({ symbol: 'TVC:GOLD', interval: '60' });
  const liveBtn = document.querySelector('.btn-period[data-period="LIVE"]');
  const url = `/chart/${window.currentPeriod}/All.json?t=${Date.now()}`;
  if (liveBtn) { document.querySelectorAll('.btn-period').forEach(b=>b.classList.remove('active')); liveBtn.classList.add('active'); }

  // init
  updateLocalTime();
  fetchGoldPrice();
  if (window.currentPeriod !== 'LIVE') {
    renderCurrentWithButtonRange();
  }

  setInterval(()=>{ updateLocalTime(); fetchGoldPrice(); }, 10000);
});