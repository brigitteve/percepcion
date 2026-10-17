"""Sesiones de audio acotadas y análisis acústico en tiempo real e histórico.

Implementa la arquitectura acústica de `final.py` (UPAO):
- Detección en vivo de pisadas y cadencia (caminando vs corriendo).
- Detección de impacto súbito de caída (pico RMS y factor de cresta).
- Detección de gritos y pedidos de auxilio (frecuencia fundamental F0 con pYIN).
- Registro persistente de alertas en `alertas.jsonl` y visualización inmediata.
- Soporte para análisis de archivos de audio/video y sesión sintética de prueba.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Blueprint, jsonify, request

from acoustic_perception import (
    SR,
    ClassificationThresholds,
    SensitivityConfig,
    StreamingAcousticDetector,
    analyze_session_full,
    build_simulated_session,
    extraer_audio_archivo,
    mejorar_audio,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MESSAGE = 'ALERTA: Se ha detectado un sonido acústico de riesgo en escaleras.'


class AudioEventStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def read(self):
        with self.lock:
            if not self.path.exists():
                return []
            rows = []
            for line in self.path.read_text(encoding='utf-8', errors='replace').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
            return rows

    def append(self, event):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')

    def delete(self, ids=None):
        with self.lock:
            rows = self.read()
            keep = [r for r in rows if ids is not None and r.get('alert_id') not in ids]
            if self.path.exists():
                self.path.write_text(
                    ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in keep),
                    encoding='utf-8',
                )
            return len(rows) - len(keep)


class AudioSession:
    def __init__(self, detector_factory, store, idle_timeout=15, sensitivity="media"):
        self.id = uuid.uuid4().hex
        self.store = store
        self.idle_timeout = idle_timeout
        self.sensitivity = sensitivity
        self.last_seen = time.monotonic()
        self.stopped = threading.Event()
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=4)
        self.state = 'iniciando'
        self.error = None
        self.last_sequence = -1
        self.processed = 0
        self.dropped = 0
        self.processing_ms = 0.0
        self.rms = 0.0
        self.live_analysis = None
        self.last_event_offset = -float('inf')
        self.worker = threading.Thread(
            target=self._run, args=(detector_factory,), daemon=True, name=f'audio-{self.id[:8]}'
        )
        self.worker.start()

    def submit(self, sequence, offset, samples):
        with self.lock:
            if self.stopped.is_set():
                raise KeyError(self.id)
            if sequence <= self.last_sequence:
                raise ValueError('Secuencia duplicada o fuera de orden')
            self.last_sequence = sequence
            self.last_seen = time.monotonic()
            try:
                self.queue.put_nowait((sequence, offset, samples))
            except queue.Full:
                self.dropped += 1
                return False
            return True

    def snapshot(self):
        with self.lock:
            snap = dict(
                stream_id=self.id,
                state=self.state,
                error=self.error,
                processed_chunks=self.processed,
                dropped_chunks=self.dropped,
                processing_ms=round(self.processing_ms, 2),
                rms=round(self.rms, 6),
            )
            if self.live_analysis:
                snap['live_analysis'] = self.live_analysis
            return snap

    def stop(self):
        with self.lock:
            self.stopped.set()
            self.state = 'detenido'
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass

    def _run(self, factory):
        detector = None
        try:
            if callable(factory):
                try:
                    detector = factory(self.sensitivity)
                except TypeError:
                    detector = factory()
            else:
                detector = factory
            expected = None
            origin = 0.0
            with self.lock:
                if not self.stopped.is_set():
                    self.state = 'esperando_audio'
            while not self.stopped.is_set():
                try:
                    item = self.queue.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() - self.last_seen > self.idle_timeout:
                        self.stop()
                    continue
                if item is None:
                    break
                seq, offset, samples = item
                if expected is None or offset != expected:
                    detector.reset()
                    origin = offset / 16000
                expected = offset + len(samples)
                started = time.perf_counter()
                events = detector.process(samples)
                with self.lock:
                    if self.stopped.is_set():
                        break
                    self.processed += 1
                    self.processing_ms = (time.perf_counter() - started) * 1000
                    self.rms = float(np.sqrt(np.mean(samples ** 2)))
                    
                    # Estado en vivo y último análisis
                    if hasattr(detector, 'last_result') and detector.last_result:
                        self.live_analysis = detector.last_result
                        self.state = detector.last_result.get('category', 'analizando')
                    else:
                        self.state = 'silencio' if self.rms < 1e-4 else 'analizando'

                    for result in events:
                        impact_offset = origin + result.get('impact_offset_s', 0.0)
                        # Cooldown general de 2.0s por alerta en la misma sesión
                        if impact_offset - self.last_event_offset < 2.0:
                            continue
                        event_id = 'audio-' + uuid.uuid4().hex
                        raw_cat = result.get('type') or result.get('predicted_class') or 'caida'
                        category = 'posible_caida' if raw_cat in ('caida', 'posible_caida') else raw_cat
                        severity = result.get('severity') or ('ALTO' if raw_cat in ('caida', 'posible_caida', 'pedido_auxilio') else 'MEDIO')
                        
                        desc_map = {
                            'posible_caida': '¡ALERTA CRÍTICA: Impacto acústico súbito compatible con caída en escaleras!',
                            'caida': '¡ALERTA CRÍTICA: Impacto acústico súbito compatible con caída en escaleras!',
                            'pedido_auxilio': '¡ALERTA CRÍTICA: Grito o pedido de auxilio detectado!',
                            'corriendo': 'ADVERTENCIA: Alta frecuencia de pisadas; persona corriendo en escaleras.',
                            'caminando': 'Tránsito peatonal seguro: persona caminando normalmente.',
                        }
                        desc = result.get('description') or desc_map.get(raw_cat, DEFAULT_MESSAGE)
                        
                        self.store.append(dict(
                            alert_id=event_id,
                            event_id=event_id,
                            type=category,
                            risk_type=result.get('risk_type', f'{category}_acustica'),
                            source='audio',
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            stream_id=self.id,
                            severity=severity,
                            description=desc,
                            impact_offset_s=round(impact_offset, 3),
                            risk_ref=result.get('risk_ref', 'Módulo de Percepción de Sonido en Escaleras (final.py)'),
                            alarm_activated=result.get('alarm_activated', severity == 'ALTO'),
                            technical=result.get('technical', {}),
                            imagen=None,
                        ))
                        self.last_event_offset = impact_offset
        except Exception as exc:
            LOGGER.exception('Fallo en sesión acústica %s', self.id)
            with self.lock:
                if not self.stopped.is_set():
                    self.state = 'error'
                    self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if detector is not None and hasattr(detector, 'reset'):
                detector.reset()


class AudioSessionManager:
    def __init__(self, store, detector_factory=None, max_sessions=4, idle_timeout=15, default_sensitivity="media"):
        self.store = store
        self.default_sensitivity = default_sensitivity
        self.detector_factory = detector_factory or self._make_detector
        self.sessions = {}
        self.lock = threading.Lock()
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout
        self.closed = threading.Event()
        self.reaper = None

    def _make_detector(self, sensitivity=None):
        sens = sensitivity or self.default_sensitivity
        return StreamingAcousticDetector(sensitivity=sens)

    def warmup(self):
        """Precalienta kernels JIT de librosa (onsets y pYIN) para cold start instantáneo."""
        LOGGER.info('Precalentando kernels acústicos...')
        detector = StreamingAcousticDetector(sensitivity=self.default_sensitivity)
        silence = np.zeros(8000, dtype=np.float32)
        detector.process(silence)
        tone = (0.05 * np.sin(2 * np.pi * 500 * np.arange(8000) / 16000)).astype(np.float32)
        detector.process(tone)
        detector.reset()
        LOGGER.info('Kernels acústicos precalentados correctamente.')

    def _reap(self):
        while not self.closed.wait(2):
            with self.lock:
                for key, value in list(self.sessions.items()):
                    if value.stopped.is_set() or time.monotonic() - value.last_seen > self.idle_timeout:
                        value.stop()
                        del self.sessions[key]

    def create(self, sensitivity=None):
        with self.lock:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError('Límite de sesiones de audio alcanzado')
            if self.reaper is None:
                self.reaper = threading.Thread(target=self._reap, daemon=True, name='audio-cleanup')
                self.reaper.start()
            sens = sensitivity or self.default_sensitivity
            session = AudioSession(self.detector_factory, self.store, self.idle_timeout, sensitivity=sens)
            self.sessions[session.id] = session
            return session

    def get(self, key):
        with self.lock:
            return self.sessions[key]

    def stop(self, key):
        with self.lock:
            session = self.sessions.pop(key, None)
        if session:
            session.stop()

    def close(self):
        self.closed.set()
        with self.lock:
            for session in self.sessions.values():
                session.stop()
            self.sessions.clear()


def audio_blueprint(manager):
    bp = Blueprint('audio_live', __name__)

    @bp.post('/api/audio/sessions')
    def create():
        try:
            data = request.get_json(silent=True) or {}
            sensitivity = data.get('sensitivity') or request.args.get('sensitivity') or manager.default_sensitivity
            return jsonify(manager.create(sensitivity=sensitivity).snapshot()), 201
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 429

    @bp.get('/api/audio/sessions/<key>')
    def status(key):
        try:
            return jsonify(manager.get(key).snapshot())
        except KeyError:
            return jsonify(error='Sesión finalizada'), 404

    @bp.post('/api/audio/sessions/<key>/chunks')
    def chunk(key):
        if request.content_length != 8000 * 4:
            return jsonify(error='Se requieren 8000 muestras float32 LE mono a 16000 Hz'), 400
        try:
            seq = int(request.headers.get('X-Audio-Sequence', ''))
            offset = int(request.headers.get('X-Audio-Offset', ''))
            if seq < 0 or offset != seq * 8000:
                raise ValueError('Secuencia u offset inválidos')
            samples = np.frombuffer(request.get_data(), dtype='<f4').copy()
            if not np.all(np.isfinite(samples)) or np.max(np.abs(samples)) > 1.05:
                raise ValueError('PCM inválido')
            session = manager.get(key)
            if session.snapshot()['state'] == 'error':
                return jsonify(session.snapshot()), 503
            accepted = session.submit(seq, offset, samples)
            return jsonify(session.snapshot()), 202 if accepted else 429
        except KeyError:
            return jsonify(error='Sesión finalizada'), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @bp.post('/api/audio/sessions/<key>/stop')
    def stop(key):
        manager.stop(key)
        return jsonify(state='detenido')

    @bp.post('/api/audio/sensitivity')
    def set_sensitivity():
        data = request.get_json(silent=True) or {}
        sens = data.get('sensitivity', 'media')
        if sens in ('alta', 'media', 'baja'):
            manager.default_sensitivity = sens
            return jsonify(ok=True, sensitivity=sens)
        return jsonify(error='Sensibilidad no válida (debe ser alta, media o baja)'), 400

    @bp.post('/api/audio/simulate')
    def simulate():
        """Genera y analiza la sesión sintética de final.py, registrando las alertas."""
        try:
            data = request.get_json(silent=True) or {}
            sens = data.get('sensitivity', 'media')
            audio, ground_truth = build_simulated_session()
            events, _ = analyze_session_full(audio, sr=SR, sensitivity=sens)
            
            # Registrar alertas de riesgo en el almacén
            now_iso = datetime.now(timezone.utc).isoformat()
            alerts_added = 0
            for ev in events:
                if ev['risk_level'] in ('ALTO', 'MEDIO'):
                    alert_id = 'audio-sim-' + uuid.uuid4().hex[:10]
                    manager.store.append(dict(
                        alert_id=alert_id,
                        event_id=alert_id,
                        type=ev['category'],
                        risk_type=f"{ev['category']}_acustica",
                        source='audio',
                        timestamp=now_iso,
                        stream_id='simulacion',
                        severity=ev['risk_level'],
                        description=ev['description'],
                        impact_offset_s=ev['start'],
                        risk_ref=f"Simulación UPAO final.py · [{ev['start']}s - {ev['end']}s]",
                        alarm_activated=ev['alarm_activated'],
                        technical=ev['metrics'],
                        imagen=None,
                    ))
                    alerts_added += 1

            return jsonify({
                "ok": True,
                "duration_s": round(len(audio) / SR, 2),
                "events": events,
                "ground_truth": ground_truth,
                "alerts_registered": alerts_added,
            })
        except Exception as exc:
            LOGGER.exception("Error en simulación")
            return jsonify(error=str(exc)), 500

    @bp.post('/api/audio/analyze-file')
    def analyze_file():
        """Analiza un archivo de audio o video subido según el pipeline de final.py."""
        if 'file' not in request.files:
            return jsonify(error='No se envió ningún archivo'), 400
        file = request.files['file']
        if not file.filename:
            return jsonify(error='Nombre de archivo vacío'), 400

        sens = request.form.get('sensitivity', 'media')
        improve = request.form.get('improve_audio', 'true').lower() in ('true', '1', 'yes')

        suffix = Path(file.filename).suffix or '.wav'
        temp_f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            file.save(temp_f.name)
            temp_f.close()
            
            audio, duration_s = extraer_audio_archivo(temp_f.name, sr=SR)
            if improve:
                audio = mejorar_audio(audio, sr=SR)

            events, _ = analyze_session_full(audio, sr=SR, sensitivity=sens)

            # Registrar alertas de riesgo
            now_iso = datetime.now(timezone.utc).isoformat()
            alerts_added = 0
            for ev in events:
                if ev['risk_level'] in ('ALTO', 'MEDIO'):
                    alert_id = 'audio-file-' + uuid.uuid4().hex[:10]
                    manager.store.append(dict(
                        alert_id=alert_id,
                        event_id=alert_id,
                        type=ev['category'],
                        risk_type=f"{ev['category']}_acustica",
                        source='audio',
                        timestamp=now_iso,
                        stream_id=Path(file.filename).name[:16],
                        severity=ev['risk_level'],
                        description=ev['description'],
                        impact_offset_s=ev['start'],
                        risk_ref=f"Archivo {Path(file.filename).name} · [{ev['start']}s - {ev['end']}s]",
                        alarm_activated=ev['alarm_activated'],
                        technical=ev['metrics'],
                        imagen=None,
                    ))
                    alerts_added += 1

            return jsonify({
                "ok": True,
                "filename": file.filename,
                "duration_s": round(duration_s, 2),
                "events": events,
                "alerts_registered": alerts_added,
            })
        except Exception as exc:
            LOGGER.exception("Error analizando archivo de audio")
            return jsonify(error=str(exc)), 500
        finally:
            try:
                os.remove(temp_f.name)
            except Exception:
                pass

    return bp


def install_audio(app, root):
    store = AudioEventStore(Path(root) / 'runs' / 'audio_alerts' / 'alertas.jsonl')
    manager = AudioSessionManager(store)
    app.register_blueprint(audio_blueprint(manager))
    atexit.register(manager.close)
    return store, manager
