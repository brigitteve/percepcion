/**
 * audio-panel.js  -  Panel dedicado "Audio en vivo" (Módulo de Percepción Acústica UPAO)
 *
 * Basado en el pipeline de final.py:
 *  - Detección en tiempo real de pisadas (caminando vs corriendo).
 *  - Detección de impacto súbito de caída (pico RMS y crest factor).
 *  - Detección de gritos y pedidos de auxilio (frecuencia fundamental F0 con pYIN).
 *  - Vúmetro RMS animado.
 *  - Alertas acústicas con badges, notificación de sistema, destello y pitido.
 *  - Análisis de archivos de audio/video y demostración simulada con ground truth.
 */

/* ------------------------------------------------------------------ helpers */
const aplQs = (sel, ctx = document) => ctx.querySelector(sel);
const aplFmt = ts => (ts || '').replace('T', ' ').replace(/\.\d+Z$/, ' UTC');

/* ------------------------------------------------------------------ beeper  */
function aplBeep(loud = false) {
  try {
    const ctx = new (globalThis.AudioContext || globalThis.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(loud ? 920 : 660, ctx.currentTime);
    if (loud) osc.frequency.linearRampToValueAtTime(480, ctx.currentTime + 0.35);
    gain.gain.setValueAtTime(loud ? 0.45 : 0.30, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + (loud ? 0.8 : 0.45));
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + (loud ? 0.8 : 0.45));
    osc.onended = () => ctx.close();
  } catch (_) {}
}

/* ------------------------------------------------------------------ VUMeter */
class AplVUMeter {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this._rms = 0;
    this._peak = 0;
    this._peakTimer = 0;
    this._raf = null;
    this._draw();
  }

  set rms(value) {
    const v = Math.min(1, Math.max(0, value));
    this._rms = v;
    if (v > this._peak) { this._peak = v; this._peakTimer = 60; }
  }

  _draw() {
    const { canvas, ctx } = this;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0d1018';
    ctx.fillRect(0, 0, W, H);

    const segments = 24;
    const segW = (W - 2) / segments;
    const rmsSegs = Math.round(this._rms * segments);
    const peakSeg = Math.round(this._peak * segments);

    for (let i = 0; i < segments; i++) {
      const x = 1 + i * segW;
      const active = i < rmsSegs;
      let color;
      if (i >= segments - 3)       color = active ? '#e03c3c' : '#2a1010';
      else if (i >= segments - 7)  color = active ? '#f2a83c' : '#251a08';
      else                         color = active ? '#3fb85a' : '#0d1f14';
      ctx.fillStyle = color;
      ctx.fillRect(x + 1, 4, segW - 2, H - 8);
      if (i === peakSeg - 1 && this._peak > 0.02) {
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.fillRect(x + 1, 4, segW - 2, H - 8);
      }
    }

    if (this._peakTimer > 0) this._peakTimer--;
    else if (this._peak > 0) this._peak = Math.max(0, this._peak - 0.01);

    this._raf = requestAnimationFrame(() => this._draw());
  }

  destroy() { if (this._raf) cancelAnimationFrame(this._raf); }
}

/* ------------------------------------------------------------------ AlertLog */
class AplAlertLog {
  constructor(container, badgeEl) {
    this.container = container;
    this.badgeEl = badgeEl;
    this._seen = new Set();
    this._count = 0;
  }

  push(alert) {
    if (this._seen.has(alert.alert_id)) return false;
    this._seen.add(alert.alert_id);
    this._count++;

    const cat = alert.type || '';
    const sev = (alert.severity || 'MEDIO').toUpperCase();
    
    // Categorización visual según final.py
    let icon = '🔴';
    let title = 'Alerta acústica';
    if (cat === 'caida' || cat === 'posible_caida') {
      icon = '💥';
      title = 'Caída / Impacto en escaleras';
    } else if (cat === 'pedido_auxilio') {
      icon = '🗣';
      title = 'Pedido de auxilio / Grito';
    } else if (cat === 'corriendo') {
      icon = '🏃';
      title = 'Persona corriendo en escaleras';
    } else if (cat === 'caminando') {
      icon = '🚶';
      title = 'Tránsito peatonal';
    }

    const clsSev = sev === 'ALTO' ? 'apl-alto' : (sev === 'MEDIO' ? 'apl-medio' : 'apl-bajo');
    const badgeCls = 'apl-badge-' + sev.toLowerCase();

    // Detalles técnicos
    let techHtml = '';
    if (alert.technical && Object.keys(alert.technical).length) {
      const parts = [];
      if (alert.technical.step_rate != null) parts.push(`Cadencia: <b>${alert.technical.step_rate}</b> pasos/s`);
      if (alert.technical.peak_rms != null) parts.push(`RMS pico: <b>${alert.technical.peak_rms}</b>`);
      if (alert.technical.crest_factor != null) parts.push(`Crest factor: <b>${alert.technical.crest_factor}</b>`);
      if (alert.technical.f0_median != null && alert.technical.f0_median > 0) parts.push(`F0: <b>${alert.technical.f0_median} Hz</b>`);
      if (alert.technical.confidence != null) parts.push(`Conf: <b>${Math.round(alert.technical.confidence * 100)}%</b>`);
      if (parts.length) {
        techHtml = `<div class="apl-meta" style="color:#8ca0c0;margin-top:2px">${parts.join(' &middot; ')}</div>`;
      }
    }

    const item = document.createElement('div');
    item.className = 'apl-item ' + clsSev;
    item.dataset.id = alert.alert_id;
    item.innerHTML = `
      <div class="apl-icon">${icon}</div>
      <div class="apl-body">
        <div class="apl-title">${title}
          <span class="apl-badge ${badgeCls}">${sev}</span>
        </div>
        <div class="apl-meta">
          ${aplFmt(alert.timestamp)}
          &middot; sesión <code>${(alert.stream_id || '').slice(0, 10)}</code>
          &middot; t = ${alert.impact_offset_s != null ? alert.impact_offset_s + 's' : '&mdash;'}
        </div>
        ${alert.description ? '<div class="apl-desc">' + alert.description + '</div>' : ''}
        ${techHtml}
        ${alert.risk_ref ? '<div class="apl-ref">' + alert.risk_ref + '</div>' : ''}
      </div>
      <button class="apl-del" data-del="${alert.alert_id}" title="Borrar esta alerta">&times;</button>`;
    this.container.prepend(item);

    const ph = aplQs('.apl-placeholder', this.container);
    if (ph) ph.remove();

    this._updateBadge();
    return true;
  }

  remove(id) {
    this._seen.delete(id);
    const el = this.container.querySelector('[data-id="' + id + '"]');
    if (el) el.remove();
    this._count = Math.max(0, this._count - 1);
    if (!this.container.querySelector('.apl-item')) {
      this.container.innerHTML = '<div class="apl-placeholder">Sin alertas acústicas registradas.</div>';
    }
    this._updateBadge();
  }

  clearAll() {
    this._seen.clear();
    this._count = 0;
    this.container.innerHTML = '<div class="apl-placeholder">Sin alertas acústicas registradas.</div>';
    this._updateBadge();
  }

  _updateBadge() {
    if (!this.badgeEl) return;
    if (this._count > 0) {
      this.badgeEl.textContent = this._count;
      this.badgeEl.style.display = 'inline-block';
    } else {
      this.badgeEl.style.display = 'none';
    }
  }
}

/* ================================================================== AudioPanel */
class AudioPanel {
  constructor() {
    this._statusEl      = aplQs('#apl-status');
    this._rmsEl         = aplQs('#apl-rms-value');
    this._dbEl          = aplQs('#apl-db-value');
    this._stateChip     = aplQs('#apl-state-chip');
    this._riskBadge     = aplQs('#apl-risk-badge');
    this._stepsVal      = aplQs('#apl-steps-val');
    this._crestVal      = aplQs('#apl-crest-val');
    this._f0Val         = aplQs('#apl-f0-val');
    this._peakRmsVal    = aplQs('#apl-peak-rms-val');
    this._vuCanvas      = aplQs('#apl-vu');
    this._logEl         = aplQs('#apl-log');
    this._badgeEl       = aplQs('#apl-tab-badge');
    this._beepChk       = aplQs('#apl-beep');
    this._notifBtn      = aplQs('#apl-notif-btn');
    this._startBtn      = aplQs('#apl-start');
    this._stopBtn       = aplQs('#apl-stop');
    this._clearBtn      = aplQs('#apl-clear');
    this._flashEl       = aplQs('#apl-flash');
    this._chunksEl      = aplQs('#apl-chunks');
    this._droppedEl     = aplQs('#apl-dropped');
    this._msEl          = aplQs('#apl-ms');
    this._sensSelect    = aplQs('#apl-sensitivity');
    this._simBtn        = aplQs('#apl-btn-sim');
    this._fileInput     = aplQs('#apl-file-input');
    this._fileImproveChk= aplQs('#apl-file-improve');
    this._fileBtn       = aplQs('#apl-file-btn');
    this._fileResultsBox= aplQs('#apl-file-results');
    this._frTitle       = aplQs('#apl-fr-title');
    this._frMeta        = aplQs('#apl-fr-meta');
    this._frTimeline    = aplQs('#apl-fr-timeline');

    this._vu    = new AplVUMeter(this._vuCanvas);
    this._log   = new AplAlertLog(this._logEl, this._badgeEl);
    this._audio = null;
    this._stream = null;
    this._vuCtx  = null;

    this._pollInterval = setInterval(() => this._pollAlerts(), 2000);
    this._bindControls();
    this._setChip('detenido');
    this._pollAlerts();
  }

  _bindControls() {
    this._startBtn.onclick = () => this._connect();
    this._stopBtn.onclick  = () => this._disconnect();
    this._clearBtn.onclick = () => this._deleteAll();

    if (this._sensSelect) {
      this._sensSelect.onchange = async () => {
        const val = this._sensSelect.value;
        try {
          await fetch('/api/audio/sensitivity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sensitivity: val }),
          });
        } catch (_) {}
      };
    }

    if (this._simBtn) {
      this._simBtn.onclick = () => this._runSimulation();
    }

    if (this._fileBtn) {
      this._fileBtn.onclick = () => this._analyzeFile();
    }

    this._notifBtn.onclick = async () => {
      if (!('Notification' in window)) { alert('Este navegador no soporta notificaciones del sistema.'); return; }
      const perm = await Notification.requestPermission();
      this._notifBtn.textContent = perm === 'granted' ? '✅ Notificaciones activas' : '❌ Permiso denegado';
    };

    this._logEl.addEventListener('click', ev => {
      const btn = ev.target.closest('[data-del]');
      if (btn) this._deleteOne(btn.dataset.del);
    });
  }

  _setChip(state, extra) {
    const map = {
      detenido:        ['chip-off',          '⏹ Detenido'],
      conectando:      ['chip-wait',         '⏳ Conectando micrófono…'],
      iniciando:       ['chip-wait',         '⏳ Iniciando motor acústico…'],
      esperando_audio: ['chip-wait',         '🎙 Esperando audio'],
      analizando:      ['chip-ok',           '🔊 Analizando en vivo'],
      caminando:       ['chip-caminando',    '🚶 Caminando'],
      corriendo:       ['chip-medio',        '🏃 Corriendo en escaleras'],
      caida:           ['chip-alto',         '💥 ¡Impacto súbito / Caída!'],
      pedido_auxilio:  ['chip-alto',         '🗣 ¡Grito / Pedido de auxilio!'],
      ruido_normal:    ['chip-ruido_normal', '🔊 Ambiente normal'],
      silencio:        ['chip-muted',        '🔇 Silencio / Sin señal'],
      error:           ['chip-error',        '⚠ Error'],
    };
    const [cls, label] = map[state] || ['chip-off', state];
    this._stateChip.className = 'apl-chip ' + cls;
    this._stateChip.textContent = label + (extra ? ' – ' + extra : '');
  }

  _updateLiveMetrics(analysis) {
    if (!analysis) return;
    
    // Cadencia de pisadas
    if (this._stepsVal) {
      this._stepsVal.textContent = (analysis.step_rate != null && analysis.step_rate > 0)
        ? `${analysis.step_rate} pasos/s`
        : '0.0';
    }

    // Factor de cresta
    if (this._crestVal) {
      this._crestVal.textContent = analysis.crest_factor != null ? analysis.crest_factor : '—';
    }

    // F0 vocal
    if (this._f0Val) {
      this._f0Val.textContent = (analysis.f0_median != null && analysis.f0_median > 0)
        ? `${analysis.f0_median} Hz`
        : '—';
    }

    // Pico RMS
    if (this._peakRmsVal) {
      this._peakRmsVal.textContent = analysis.peak_rms != null ? analysis.peak_rms : '—';
    }

    // Nivel de riesgo badge
    if (this._riskBadge && analysis.risk_level) {
      const rl = analysis.risk_level.toUpperCase();
      this._riskBadge.className = 'apl-badge apl-badge-' + rl.toLowerCase();
      this._riskBadge.textContent = 'RIESGO ' + rl;
    }

    // Estado del chip
    if (analysis.category) {
      this._setChip(analysis.category);
    }

    // Si se activa alarma
    if (analysis.alarm_activated) {
      this._flash(analysis.description || '¡ALERTA CRÍTICA ACÚSTICA DETECTADA!');
      if (this._beepChk && this._beepChk.checked) aplBeep(true);
    }
  }

  async _connect() {
    if (this._stream) this._disconnect();
    this._setChip('conectando');
    this._startBtn.disabled = true;
    this._stopBtn.disabled  = false;
    this._statusEl.textContent = 'Solicitando acceso al micrófono…';

    try {
      this._stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        video: false,
      });
    } catch (e) {
      const msgs = {
        NotAllowedError:  'Permiso de micrófono denegado.',
        NotFoundError:    'No se encontró ningún micrófono.',
        NotReadableError: 'El micrófono está en uso por otra aplicación.',
      };
      this._statusEl.textContent = msgs[e.name] || ('Error: ' + e.message);
      this._setChip('error');
      this._startBtn.disabled = false;
      this._stopBtn.disabled  = true;
      return;
    }

    this._connectVU(this._stream);

    this._audio = new LiveFallAudio(
      (msg, data) => {
        this._statusEl.textContent = msg;
        if (data && data.live_analysis) {
          this._updateLiveMetrics(data.live_analysis);
        }
      },
      data => {
        if (data && data.live_analysis) {
          this._updateLiveMetrics(data.live_analysis);
        }
      }
    );

    this._audio.start(this._stream);
    this._setChip('iniciando');
  }

  _disconnect() {
    if (this._audio)  { this._audio.stop(); this._audio = null; }
    if (this._vuCtx)  { this._vuCtx.close().catch(() => {}); this._vuCtx = null; }
    if (this._stream) { this._stream.getTracks().forEach(t => t.stop()); this._stream = null; }
    this._startBtn.disabled = false;
    this._stopBtn.disabled  = true;
    this._rmsEl.textContent = '0.0000';
    this._dbEl.textContent  = '—';
    this._vu.rms = 0;
    this._setChip('detenido');
    this._statusEl.textContent = 'Análisis acústico detenido.';
    if (this._stepsVal)   this._stepsVal.textContent = '—';
    if (this._crestVal)   this._crestVal.textContent = '—';
    if (this._f0Val)      this._f0Val.textContent = '—';
    if (this._peakRmsVal) this._peakRmsVal.textContent = '—';
    if (this._riskBadge) {
      this._riskBadge.className = 'apl-badge apl-badge-bajo';
      this._riskBadge.textContent = 'RIESGO BAJO';
    }
  }

  _connectVU(stream) {
    try {
      const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
      this._vuCtx = new Ctx();
      const src = this._vuCtx.createMediaStreamSource(stream);
      const ana = this._vuCtx.createAnalyser();
      ana.fftSize = 256;
      src.connect(ana);
      const buf = new Float32Array(ana.fftSize);
      const tick = () => {
        if (!this._vuCtx) return;
        ana.getFloatTimeDomainData(buf);
        let sum = 0;
        for (const s of buf) sum += s * s;
        const rms = Math.sqrt(sum / buf.length);
        this._vu.rms = rms * 4;
        const db = rms > 1e-6 ? (20 * Math.log10(rms)).toFixed(1) : null;
        this._rmsEl.textContent = rms.toFixed(4);
        this._dbEl.textContent  = db !== null ? db + ' dB' : '—';
        requestAnimationFrame(tick);
      };
      tick();
    } catch (_) {}
  }

  async _pollAlerts() {
    try {
      const resp = await fetch('/api/alerts');
      if (!resp.ok) return;
      const data = await resp.json();
      const lista = (data.alertas || []).filter(a => a.source === 'audio');
      for (const a of lista) {
        if (this._log.push(a)) {
          this._flash(a.description || '¡Alerta acústica detectada!');
          if (this._beepChk && this._beepChk.checked) aplBeep(a.severity === 'ALTO');
          this._notify(a);
          const audioTabActive = document.getElementById('p-audio') &&
                                 document.getElementById('p-audio').classList.contains('on');
          if (!audioTabActive && this._badgeEl) {
            this._badgeEl.style.display = 'inline-block';
            this._badgeEl.textContent = this._log._count;
          }
        }
      }
      if (this._audio && this._audio.session) {
        try {
          const sr = await fetch('/api/audio/sessions/' + this._audio.session);
          if (sr.ok) {
            const sd = await sr.json();
            if (this._chunksEl)  this._chunksEl.textContent  = sd.processed_chunks  != null ? sd.processed_chunks  : '—';
            if (this._droppedEl) this._droppedEl.textContent = sd.dropped_chunks     != null ? sd.dropped_chunks     : '—';
            if (this._msEl)      this._msEl.textContent      = sd.processing_ms      != null ? sd.processing_ms + ' ms' : '—';
            if (sd.live_analysis) {
              this._updateLiveMetrics(sd.live_analysis);
            }
          }
        } catch (_) {}
      }
    } catch (_) {}
  }

  _flash(text = 'ALERTA ACÚSTICA') {
    if (!this._flashEl) return;
    const txtSpan = this._flashEl.querySelector('span:nth-child(2)');
    if (txtSpan) txtSpan.textContent = text;
    this._flashEl.style.display = 'flex';
    this._flashEl.classList.remove('apl-flash-anim');
    void this._flashEl.offsetWidth;
    this._flashEl.classList.add('apl-flash-anim');
    setTimeout(() => { if (this._flashEl) this._flashEl.style.display = 'none'; }, 4500);
  }

  _notify(alert) {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    try {
      new Notification('Alerta acústica en escaleras', {
        body: `${alert.severity}: ${alert.description || 'Sonido detectado'}`,
        icon: '/static/favicon.ico',
      });
    } catch (_) {}
  }

  async _deleteOne(id) {
    try {
      await fetch('/api/alertas?id=' + encodeURIComponent(id), { method: 'DELETE' });
      this._log.remove(id);
    } catch (e) { alert('No se pudo borrar la alerta: ' + e.message); }
  }

  async _deleteAll() {
    if (!confirm('¿Borrar todas las alertas acústicas?')) return;
    try {
      await fetch('/api/alertas', { method: 'DELETE' });
      this._log.clearAll();
    } catch (e) { alert('No se pudieron borrar las alertas: ' + e.message); }
  }

  /* ------------------------------------------------ Demostración y Archivos */
  async _runSimulation() {
    this._simBtn.disabled = true;
    this._simBtn.textContent = '⏳ Ejecutando simulación UPAO…';
    try {
      const sens = this._sensSelect ? this._sensSelect.value : 'media';
      const resp = await fetch('/api/audio/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sensitivity: sens }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Error al ejecutar simulación');
      this._renderFileResults(data, '⚡ Demostración Simulada UPAO (final.py)');
      this._pollAlerts();
    } catch (e) {
      alert('Error en simulación: ' + e.message);
    } finally {
      this._simBtn.disabled = false;
      this._simBtn.textContent = '⚡ Cargar sesión simulada (Demo UPAO)';
    }
  }

  async _analyzeFile() {
    if (!this._fileInput || !this._fileInput.files.length) {
      alert('Por favor selecciona un archivo de audio o video primero.');
      return;
    }
    const file = this._fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('sensitivity', this._sensSelect ? this._sensSelect.value : 'media');
    formData.append('improve_audio', this._fileImproveChk && this._fileImproveChk.checked ? 'true' : 'false');

    this._fileBtn.disabled = true;
    this._fileBtn.textContent = '⏳ Analizando archivo…';
    try {
      const resp = await fetch('/api/audio/analyze-file', {
        method: 'POST',
        body: formData,
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Error analizando archivo');
      this._renderFileResults(data, `📁 Archivo: ${file.name}`);
      this._pollAlerts();
    } catch (e) {
      alert('Error: ' + e.message);
    } finally {
      this._fileBtn.disabled = false;
      this._fileBtn.textContent = '🚀 Analizar archivo';
    }
  }

  _renderFileResults(data, title) {
    if (!this._fileResultsBox || !this._frTimeline) return;
    this._fileResultsBox.style.display = 'block';
    if (this._frTitle) this._frTitle.textContent = title;
    if (this._frMeta) {
      this._frMeta.textContent = `Duración: ${data.duration_s}s · Eventos: ${data.events.length} · Alertas añadidas: ${data.alerts_registered}`;
    }

    if (!data.events.length) {
      this._frTimeline.innerHTML = '<div style="color:var(--muted);text-align:center;padding:12px">No se detectaron eventos de riesgo en este audio.</div>';
      return;
    }

    const catIcons = {
      caida: '💥',
      pedido_auxilio: '🗣',
      corriendo: '🏃',
      caminando: '🚶',
      ruido_normal: '🔊',
    };

    let html = '';
    for (const ev of data.events) {
      const sev = (ev.risk_level || 'BAJO').toUpperCase();
      const badgeCls = 'apl-badge-' + sev.toLowerCase();
      const icon = catIcons[ev.category] || '📌';
      
      let metricStr = '';
      if (ev.metrics) {
        if (ev.category === 'corriendo' || ev.category === 'caminando') {
          metricStr = `&middot; ${ev.metrics.step_rate || 0} pasos/s`;
        } else if (ev.category === 'caida') {
          metricStr = `&middot; RMS pico: ${ev.metrics.peak_rms || 0} (Crest: ${ev.metrics.crest_factor || 0})`;
        } else if (ev.category === 'pedido_auxilio') {
          metricStr = `&middot; F0: ${ev.metrics.f0_median || 0} Hz`;
        }
      }

      html += `
        <div class="apl-tl-item">
          <div>
            <span class="apl-tl-time">[${ev.start}s - ${ev.end}s]</span>
            <span>${icon} <b>${ev.category_label || ev.category}</b></span>
            <span style="font-size:11px;color:var(--muted);margin-left:6px">${metricStr}</span>
          </div>
          <div>
            <span class="apl-badge ${badgeCls}">${sev}</span>
            <span style="font-size:11px;color:var(--muted);margin-left:6px">${Math.round(ev.confidence * 100)}% conf</span>
          </div>
        </div>`;
    }
    this._frTimeline.innerHTML = html;
  }
}

/* ------------------------------------------------------------------ auto init */
let _aplInstance = null;
function initAudioPanel() {
  if (!_aplInstance && aplQs('#p-audio')) {
    _aplInstance = new AudioPanel();
  }
  return _aplInstance;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initAudioPanel);
} else {
  initAudioPanel();
}
