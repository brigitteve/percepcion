# -*- coding: utf-8 -*-
"""Módulo de Percepción de Sonido en Escaleras — Sistema de Alerta ante Caídas y Auxilio.

Implementa el pipeline acústico de `final.py` (UPAO):
1. Filtro pasa-altos (Butterworth 80 Hz) para eliminar rumor grave y offset DC.
2. Estimación de cadencia de pisadas (onsets con librosa) para detectar 'caminando' y 'corriendo'.
3. Detección de impacto súbito (energía RMS + crest factor) para detectar 'caida'.
4. Estimación de frecuencia fundamental F0 (librosa.pyin) para detectar 'pedido_auxilio' (gritos).
5. Motor de clasificación por umbrales con sensibilidad configurable (alta, media, baja).
6. Análisis tanto en tiempo real (streaming continuo) como en archivos completos (audio/video).
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except Exception:
    HAS_NOISEREDUCE = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    HAS_SOUNDFILE = False

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = shutil.which("ffmpeg")

LOGGER = logging.getLogger(__name__)

SR = 16000  # Frecuencia estándar para streaming y análisis acústico
CORTE_PASA_ALTOS_HZ = 80.0


@dataclass
class ClassificationThresholds:
    running_steps_per_sec: float = 2.8      # Cadencia mínima para considerar 'corriendo' (pasos/s)
    walking_steps_min: float = 0.7          # Cadencia mínima para considerar 'caminando'
    scream_fmin: float = 240.0              # Frecuencia fundamental F0 mínima de grito (Hz)
    scream_fmax: float = 950.0              # Frecuencia fundamental F0 máxima de grito (Hz)
    scream_min_confidence: float = 0.40     # Confianza mínima de voz (voiced_prob)
    impact_rms_threshold: float = 0.22      # Pico RMS mínimo para impacto súbito
    impact_crest_factor: float = 5.0        # Factor de cresta mínimo (peak / mean RMS)


class SensitivityConfig:
    """Ajuste de sensibilidad (RF-07 en final.py)."""
    _SCALES = {"baja": 1.25, "media": 1.0, "alta": 0.78}

    def __init__(self, level: str = "media"):
        if level not in self._SCALES:
            level = "media"
        self.level = level
        self.scale = self._SCALES[level]

    def apply(self, base: Optional[ClassificationThresholds] = None) -> ClassificationThresholds:
        base = base or ClassificationThresholds()
        adjusted = ClassificationThresholds(**asdict(base))
        adjusted.running_steps_per_sec *= self.scale
        adjusted.walking_steps_min *= self.scale
        adjusted.impact_rms_threshold *= self.scale
        adjusted.impact_crest_factor *= self.scale
        adjusted.scream_min_confidence = max(0.25, min(0.90, adjusted.scream_min_confidence * self.scale))
        return adjusted


def filtro_pasa_altos(y: np.ndarray, sr: int = SR, corte_hz: float = CORTE_PASA_ALTOS_HZ) -> np.ndarray:
    """Filtro pasa-altos Butterworth (orden 4, fase cero)."""
    if len(y) < 64:
        return y.astype(np.float32)
    sos = butter(4, corte_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def compute_footstep_rate(segment: np.ndarray, sr: int = SR) -> Tuple[float, int]:
    """Calcula la cadencia de pisadas por segundo (onsets) usando librosa."""
    if len(segment) < 512:
        return 0.0, 0
    duration = len(segment) / sr
    if duration <= 0:
        return 0.0, 0
    try:
        onset_env = librosa.onset.onset_strength(y=segment, sr=sr, hop_length=512)
        # Evita contar micro-rebotes usando wait (~180 ms entre pisadas posibles)
        wait_frames = max(1, int(0.18 * sr / 512))
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=512, wait=wait_frames, delta=0.15, backtrack=False
        )
        rate = len(onsets) / duration
        return float(rate), int(len(onsets))
    except Exception as exc:
        LOGGER.debug("Error en compute_footstep_rate: %s", exc)
        return 0.0, 0


def compute_impact_score(segment: np.ndarray, sr: int = SR) -> Tuple[float, float]:
    """Calcula pico RMS y factor de cresta para detectar impactos súbitos."""
    if len(segment) < 512:
        return 0.0, 0.0
    try:
        rms_frames = librosa.feature.rms(y=segment, frame_length=512, hop_length=128)[0]
        if len(rms_frames) == 0:
            return 0.0, 0.0
        peak_rms = float(np.max(rms_frames))
        mean_rms = float(np.mean(rms_frames)) + 1e-6
        crest_factor = float(peak_rms / mean_rms)
        return peak_rms, crest_factor
    except Exception as exc:
        LOGGER.debug("Error en compute_impact_score: %s", exc)
        return 0.0, 0.0


def compute_fundamental_frequency(
    segment: np.ndarray, sr: int = SR, fmin: float = 200.0, fmax: float = 1000.0
) -> Tuple[float, float]:
    """Estima la frecuencia fundamental F0 y confianza para detectar gritos/auxilio."""
    if len(segment) < int(sr * 0.15):
        return 0.0, 0.0
    # Optimización: si la energía es muy baja, no hay voz significativa
    if float(np.sqrt(np.mean(segment ** 2))) < 0.015:
        return 0.0, 0.0
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            segment, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=256
        )
        if f0 is None or not np.any(voiced_flag):
            return 0.0, 0.0
        f0_voiced = f0[voiced_flag]
        conf_voiced = voiced_prob[voiced_flag]
        valid = ~np.isnan(f0_voiced)
        f0_valid = f0_voiced[valid]
        if len(f0_valid) == 0:
            return 0.0, 0.0
        return float(np.median(f0_valid)), float(np.mean(conf_voiced[valid]))
    except Exception as exc:
        LOGGER.debug("Error en compute_fundamental_frequency: %s", exc)
        return 0.0, 0.0


def classify_window(
    segment: np.ndarray, sr: int, thresholds: ClassificationThresholds
) -> Dict[str, any]:
    """Clasifica un segmento según la jerarquía de prioridades de final.py:
    1. Impacto / caída (Riesgo ALTO)
    2. Pedido de auxilio / grito (Riesgo ALTO)
    3. Persona corriendo (Riesgo MEDIO)
    4. Persona caminando (Riesgo BAJO)
    5. Ruido normal / silencio (Riesgo BAJO)
    """
    overall_rms = float(np.sqrt(np.mean(segment ** 2)))
    if overall_rms < 1e-4:
        return {
            "category": "silencio",
            "category_label": "Silencio",
            "risk_level": "BAJO",
            "confidence": 0.95,
            "alarm_activated": False,
            "step_rate": 0.0,
            "n_onsets": 0,
            "peak_rms": 0.0,
            "crest_factor": 1.0,
            "f0_median": 0.0,
            "f0_conf": 0.0,
            "description": "Sin señal de audio o silencio ambiental.",
        }

    step_rate, n_onsets = compute_footstep_rate(segment, sr)
    peak_rms, crest = compute_impact_score(segment, sr)
    f0_median, f0_conf = compute_fundamental_frequency(
        segment, sr, fmin=thresholds.scream_fmin - 40, fmax=thresholds.scream_fmax + 50
    )

    # 1) Impacto / Caída -> Riesgo ALTO (máxima prioridad)
    if peak_rms >= thresholds.impact_rms_threshold and crest >= thresholds.impact_crest_factor:
        confidence = min(
            0.99,
            0.5 * (peak_rms / thresholds.impact_rms_threshold)
            + 0.5 * (crest / thresholds.impact_crest_factor),
        )
        return {
            "category": "caida",
            "category_label": "Caída / Impacto en escaleras",
            "risk_level": "ALTO",
            "confidence": round(float(confidence), 2),
            "alarm_activated": True,
            "step_rate": round(step_rate, 2),
            "n_onsets": n_onsets,
            "peak_rms": round(peak_rms, 4),
            "crest_factor": round(crest, 2),
            "f0_median": round(f0_median, 1),
            "f0_conf": round(f0_conf, 2),
            "description": "¡ALERTA CRÍTICA: Impacto acústico súbito compatible con caída en escaleras!",
        }

    # 2) Pedido de auxilio / Grito -> Riesgo ALTO
    if (
        thresholds.scream_fmin <= f0_median <= thresholds.scream_fmax
        and f0_conf >= thresholds.scream_min_confidence
        and peak_rms >= 0.025
    ):
        confidence = min(0.99, f0_conf)
        return {
            "category": "pedido_auxilio",
            "category_label": "Pedido de auxilio / Grito",
            "risk_level": "ALTO",
            "confidence": round(float(confidence), 2),
            "alarm_activated": True,
            "step_rate": round(step_rate, 2),
            "n_onsets": n_onsets,
            "peak_rms": round(peak_rms, 4),
            "crest_factor": round(crest, 2),
            "f0_median": round(f0_median, 1),
            "f0_conf": round(f0_conf, 2),
            "description": "¡ALERTA CRÍTICA: Frecuencia vocal compatible con grito o pedido de auxilio!",
        }

    # 3) Persona corriendo -> Riesgo MEDIO
    if step_rate >= thresholds.running_steps_per_sec and peak_rms >= 0.015:
        confidence = min(0.95, step_rate / (thresholds.running_steps_per_sec * 1.4))
        return {
            "category": "corriendo",
            "category_label": "Persona corriendo en escaleras",
            "risk_level": "MEDIO",
            "confidence": round(float(confidence), 2),
            "alarm_activated": False,
            "step_rate": round(step_rate, 2),
            "n_onsets": n_onsets,
            "peak_rms": round(peak_rms, 4),
            "crest_factor": round(crest, 2),
            "f0_median": round(f0_median, 1),
            "f0_conf": round(f0_conf, 2),
            "description": f"ADVERTENCIA: Alta frecuencia de pisadas ({step_rate:.1f} pasos/s); persona corriendo.",
        }

    # 4) Persona caminando -> Riesgo BAJO
    if step_rate >= thresholds.walking_steps_min and peak_rms >= 0.010:
        confidence = min(0.90, max(0.50, step_rate / 2.0))
        return {
            "category": "caminando",
            "category_label": "Persona caminando",
            "risk_level": "BAJO",
            "confidence": round(float(confidence), 2),
            "alarm_activated": False,
            "step_rate": round(step_rate, 2),
            "n_onsets": n_onsets,
            "peak_rms": round(peak_rms, 4),
            "crest_factor": round(crest, 2),
            "f0_median": round(f0_median, 1),
            "f0_conf": round(f0_conf, 2),
            "description": f"Tránsito seguro: ritmo normal de pisadas ({step_rate:.1f} pasos/s).",
        }

    # 5) Ruido normal -> Riesgo BAJO
    return {
        "category": "ruido_normal",
        "category_label": "Ruido ambiental normal",
        "risk_level": "BAJO",
        "confidence": 0.85,
        "alarm_activated": False,
        "step_rate": round(step_rate, 2),
        "n_onsets": n_onsets,
        "peak_rms": round(peak_rms, 4),
        "crest_factor": round(crest, 2),
        "f0_median": round(f0_median, 1),
        "f0_conf": round(f0_conf, 2),
        "description": "Ruido ambiental normal de escaleras; sin situaciones de riesgo.",
    }


class StreamingAcousticDetector:
    """Detector adaptado para streaming continuo en bloques de 500 ms (8000 muestras a 16 kHz).
    
    Conserva una ventana deslizante de 2.0 s para analizar transitorios, cadencia y voz.
    Genera eventos y alertas de riesgo en tiempo real sin requerir pausas ni esperas de 7 s.
    """

    def __init__(self, sensitivity: str = "media"):
        self.sr = SR
        self.sensitivity = sensitivity
        self.thresholds = SensitivityConfig(sensitivity).apply()
        self.buffer_duration_sec = 2.0
        self.buffer_size = int(self.buffer_duration_sec * self.sr)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.total_samples = 0
        self.sos = butter(4, CORTE_PASA_ALTOS_HZ, btype="highpass", fs=self.sr, output="sos")
        self.filter_state = np.zeros((len(self.sos), 2))
        
        # Cooldowns para no duplicar alertas continuas del mismo evento
        self.last_impact_time = -999.0
        self.last_scream_time = -999.0
        self.last_running_time = -999.0
        self.impact_cooldown = 3.0
        self.scream_cooldown = 3.5
        self.running_cooldown = 4.0
        
        # Historial reciente de la última clasificación para el snapshot
        self.last_result: Optional[Dict[str, any]] = None

    def reset(self):
        self.buffer.fill(0)
        self.total_samples = 0
        self.filter_state = np.zeros((len(self.sos), 2))
        self.last_impact_time = -999.0
        self.last_scream_time = -999.0
        self.last_running_time = -999.0
        self.last_result = None

    def process(self, samples: np.ndarray) -> List[Dict[str, any]]:
        """Procesa un bloque entrante de muestras float32 (habitualmente 500 ms).
        
        Devuelve una lista de eventos de alerta generados (si los hay).
        """
        if len(samples) == 0:
            return []

        # Filtro pasa-altos streaming
        filtered, self.filter_state = sosfilt(self.sos, samples, zi=self.filter_state)
        
        # Desplazar buffer circular
        n = len(filtered)
        if n >= self.buffer_size:
            self.buffer = filtered[-self.buffer_size:].astype(np.float32)
        else:
            self.buffer[:-n] = self.buffer[n:]
            self.buffer[-n:] = filtered.astype(np.float32)

        self.total_samples += n
        current_time_s = self.total_samples / self.sr

        # Clasificar la ventana reciente
        res = classify_window(self.buffer, self.sr, self.thresholds)
        res["timestamp_s"] = current_time_s
        self.last_result = res

        events = []
        category = res["category"]
        risk = res["risk_level"]

        # Evaluar disparo de alertas según categoría y cooldown
        if category == "caida" and (current_time_s - self.last_impact_time >= self.impact_cooldown):
            self.last_impact_time = current_time_s
            events.append({
                "type": "caida",
                "risk_type": "caida_escaleras_impacto",
                "severity": "ALTO",
                "alarm_activated": True,
                "impact_offset_s": round(current_time_s, 2),
                "description": res["description"],
                "risk_ref": "RF-02 (2) · Sonido de impacto súbito en escaleras (energía y crest factor)",
                "technical": {
                    "peak_rms": res["peak_rms"],
                    "crest_factor": res["crest_factor"],
                    "confidence": res["confidence"],
                },
            })

        elif category == "pedido_auxilio" and (current_time_s - self.last_scream_time >= self.scream_cooldown):
            self.last_scream_time = current_time_s
            events.append({
                "type": "pedido_auxilio",
                "risk_type": "grito_auxilio",
                "severity": "ALTO",
                "alarm_activated": True,
                "impact_offset_s": round(current_time_s, 2),
                "description": res["description"],
                "risk_ref": "RF-02 (3) · Frecuencia fundamental de grito o auxilio (pYIN F0)",
                "technical": {
                    "f0_median": res["f0_median"],
                    "f0_conf": res["f0_conf"],
                    "confidence": res["confidence"],
                },
            })

        elif category == "corriendo" and (current_time_s - self.last_running_time >= self.running_cooldown):
            self.last_running_time = current_time_s
            events.append({
                "type": "corriendo",
                "risk_type": "persona_corriendo_escaleras",
                "severity": "MEDIO",
                "alarm_activated": False,
                "impact_offset_s": round(current_time_s, 2),
                "description": res["description"],
                "risk_ref": "RF-02 (1) · Alta cadencia de pisadas por segundo (onsets)",
                "technical": {
                    "step_rate": res["step_rate"],
                    "n_onsets": res["n_onsets"],
                    "confidence": res["confidence"],
                },
            })

        return events


# --------------------------------------------------------------------------- #
# Análisis de Archivos Completos (Audio o Video) según final.py
# --------------------------------------------------------------------------- #

def tramo_mas_silencioso(y: np.ndarray, sr: int, dur: float = 0.5) -> np.ndarray:
    win = int(dur * sr)
    if len(y) <= win:
        return y
    hop = win // 2
    rms = [np.sqrt(np.mean(y[i : i + win] ** 2)) for i in range(0, len(y) - win, hop)]
    if not rms:
        return y[:win]
    i = int(np.argmin(rms)) * hop
    return y[i : i + win]


def estimar_snr_db(y: np.ndarray, sr: int = SR) -> float:
    rms = librosa.feature.rms(y=y)[0] + 1e-9
    return float(20 * np.log10(np.percentile(rms, 95) / np.percentile(rms, 10)))


def mejorar_audio(y: np.ndarray, sr: int = SR, fuerza: float = 0.85) -> np.ndarray:
    """Mejora de audio con filtro pasa-altos, reducción de ruido espectral y normalización."""
    y = y - np.mean(y)
    y = filtro_pasa_altos(y, sr)
    if HAS_NOISEREDUCE and len(y) >= sr * 0.5:
        try:
            ruido = tramo_mas_silencioso(y, sr)
            y = nr.reduce_noise(y=y, sr=sr, y_noise=ruido, stationary=True, prop_decrease=fuerza, n_fft=1024)
        except Exception as exc:
            LOGGER.debug("Fallo en noisereduce: %s", exc)
    y = y.astype(np.float32)
    pico = np.max(np.abs(y)) + 1e-9
    return (y / pico * 0.9).astype(np.float32)


def extraer_audio_archivo(ruta_origen: str | Path, sr: int = SR) -> Tuple[np.ndarray, float]:
    """Carga audio de cualquier archivo de audio o video usando ffmpeg o librosa."""
    ruta = Path(ruta_origen)
    if not ruta.is_file():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")

    ffmpeg = FFMPEG_BIN
    if ffmpeg:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        try:
            cmd = [
                str(ffmpeg), "-y", "-loglevel", "error", "-i", str(ruta),
                "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", temp_wav.name
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(temp_wav.name) and os.path.getsize(temp_wav.name) > 100:
                y, _ = librosa.load(temp_wav.name, sr=sr, mono=True)
                dur = len(y) / sr
                return y.astype(np.float32), dur
        except Exception as exc:
            LOGGER.debug("ffmpeg fallo, recurriendo a librosa: %s", exc)
        finally:
            try:
                os.remove(temp_wav.name)
            except Exception:
                pass

    # Respaldo con librosa
    y, _ = librosa.load(str(ruta), sr=sr, mono=True)
    return y.astype(np.float32), len(y) / sr


def analyze_session_full(
    audio: np.ndarray,
    sr: int = SR,
    window_sec: float = 1.2,
    hop_sec: float = 0.4,
    sensitivity: str = "media",
    stair_id: str = "Escalera-Principal",
) -> Tuple[List[Dict[str, any]], ClassificationThresholds]:
    """Análisis completo de una sesión por ventana deslizante (final.py sección 8).
    
    Fusiona ventanas consecutivas de la misma categoría y devuelve los eventos detectados.
    """
    thresholds = SensitivityConfig(sensitivity).apply()
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    raw_windows = []

    t = 0
    while t + win <= len(audio):
        segment = audio[t : t + win]
        res = classify_window(segment, sr, thresholds)
        raw_windows.append((t / sr, (t + win) / sr, res))
        t += hop

    # Fusión de ventanas consecutivas de la misma categoría
    merged = []
    current = None
    for start, end, res in raw_windows:
        cat = res["category"]
        if cat in ("ruido_normal", "silencio"):
            if current:
                merged.append(current)
                current = None
            continue

        if current and current["category"] == cat and start <= current["end"] + hop_sec + 1e-4:
            current["end"] = end
            current["confidence"] = max(current["confidence"], res["confidence"])
            if res["category"] == "corriendo":
                current["step_rate"] = max(current.get("step_rate", 0), res.get("step_rate", 0))
            if res["category"] == "caida":
                current["peak_rms"] = max(current.get("peak_rms", 0), res.get("peak_rms", 0))
                current["crest_factor"] = max(current.get("crest_factor", 0), res.get("crest_factor", 0))
            if res["category"] == "pedido_auxilio":
                current["f0_median"] = res.get("f0_median", current.get("f0_median", 0))
        else:
            if current:
                merged.append(current)
            current = {
                "category": cat,
                "category_label": res["category_label"],
                "risk_level": res["risk_level"],
                "confidence": res["confidence"],
                "start": start,
                "end": end,
                "alarm_activated": res["alarm_activated"],
                "description": res["description"],
                "step_rate": res.get("step_rate", 0),
                "peak_rms": res.get("peak_rms", 0),
                "crest_factor": res.get("crest_factor", 0),
                "f0_median": res.get("f0_median", 0),
            }
    if current:
        merged.append(current)

    events = []
    for m in merged:
        events.append({
            "category": m["category"],
            "category_label": m["category_label"],
            "risk_level": m["risk_level"],
            "confidence": round(m["confidence"], 2),
            "start": round(m["start"], 2),
            "end": round(m["end"], 2),
            "duration": round(m["end"] - m["start"], 2),
            "alarm_activated": m["alarm_activated"],
            "description": m["description"],
            "stair_id": stair_id,
            "metrics": {
                "step_rate": m.get("step_rate", 0),
                "peak_rms": m.get("peak_rms", 0),
                "crest_factor": m.get("crest_factor", 0),
                "f0_median": m.get("f0_median", 0),
            },
        })

    return events, thresholds


# --------------------------------------------------------------------------- #
# Generador de Sesión Sintética para Pruebas / Demostración (final.py sec 4)
# --------------------------------------------------------------------------- #

def generate_footstep(duration: float = 0.15, freq: float = 90.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, 0.3, len(t))
    env = np.exp(-t * 25)
    return (amp * (tone * 0.6 + noise * 0.4) * env).astype(np.float32)


def generate_footstep_sequence(
    total_duration: float, steps_per_second: float, sr: int = SR, amp: float = 0.5, jitter: float = 0.02
) -> np.ndarray:
    audio = np.zeros(int(total_duration * sr), dtype=np.float32)
    step_interval = 1.0 / max(0.1, steps_per_second)
    t_cursor = 0.0
    while t_cursor < total_duration:
        jittered = max(t_cursor + np.random.uniform(-jitter, jitter), 0.0)
        idx = int(jittered * sr)
        step = generate_footstep(sr=sr, amp=amp * float(np.random.uniform(0.85, 1.15)))
        end = min(idx + len(step), len(audio))
        if end > idx:
            audio[idx:end] += step[: end - idx]
        t_cursor += step_interval
    return audio


def generate_ambient_noise(duration: float, sr: int = SR, amp: float = 0.04) -> np.ndarray:
    return np.random.normal(0, amp, int(duration * sr)).astype(np.float32)


def generate_impact(duration: float = 0.3, sr: int = SR, amp: float = 0.9) -> np.ndarray:
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)
    noise = np.random.normal(0, 1, n)
    low_thump = np.sin(2 * np.pi * 60 * t)
    env = np.exp(-t * 15)
    return (amp * (noise * 0.5 + low_thump * 0.5) * env).astype(np.float32)


def generate_scream(duration: float = 1.0, sr: int = SR, amp: float = 0.7, f0: float = 540.0) -> np.ndarray:
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)
    vibrato = 20 * np.sin(2 * np.pi * 6 * t)
    freq_track = f0 + vibrato + np.linspace(0, 80, n)
    phase = 2 * np.pi * np.cumsum(freq_track) / sr
    voice = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    noise = np.random.normal(0, 0.15, n)
    attack = int(0.05 * sr)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[-attack:] = np.linspace(1, 0, attack)
    return (amp * (voice * 0.85 + noise * 0.15) * env).astype(np.float32)


def overlay(base: np.ndarray, sound: np.ndarray, offset_sec: float = 0.0, sr: int = SR) -> np.ndarray:
    start = int(offset_sec * sr)
    end = min(start + len(sound), len(base))
    if end > start:
        base[start:end] += sound[: end - start]
    return base


def build_simulated_session() -> Tuple[np.ndarray, List[Dict[str, any]]]:
    """Genera la sesión de prueba de final.py:
    1. Caminando (3.0s) -> Riesgo BAJO
    2. Corriendo (2.5s) -> Riesgo MEDIO
    3. Silencio (0.6s)
    4. Caída con impacto súbito (1.0s) -> Riesgo ALTO
    5. Pedido de auxilio (1.6s) -> Riesgo ALTO
    6. Caminando (2.5s) -> Riesgo BAJO
    """
    segments = []
    ground_truth = []
    cursor = 0.0

    def add_segment(audio_seg, label):
        nonlocal cursor
        dur = len(audio_seg) / SR
        segments.append(audio_seg)
        ground_truth.append({"label": label, "start": round(cursor, 2), "end": round(cursor + dur, 2)})
        cursor += dur

    # 1. Caminando
    add_segment(generate_ambient_noise(3.0, amp=0.03) + generate_footstep_sequence(3.0, 1.8, amp=0.35), "caminando")
    # 2. Corriendo
    add_segment(generate_ambient_noise(2.5, amp=0.03) + generate_footstep_sequence(2.5, 4.0, amp=0.5), "corriendo")
    # 3. Pausa
    add_segment(generate_ambient_noise(0.6, amp=0.02), "ruido_normal")
    # 4. Caída / impacto
    caida = overlay(generate_ambient_noise(1.0, amp=0.02), generate_impact(duration=0.3, amp=0.95), offset_sec=0.25)
    add_segment(caida, "caida")
    # 5. Pedido de auxilio
    grito = overlay(generate_ambient_noise(1.6, amp=0.02), generate_scream(duration=1.2, amp=0.75, f0=520.0), offset_sec=0.2)
    add_segment(grito, "pedido_auxilio")
    # 6. Caminando normal
    add_segment(generate_ambient_noise(2.5, amp=0.03) + generate_footstep_sequence(2.5, 1.6, amp=0.3), "caminando")

    full = np.concatenate(segments).astype(np.float32)
    # Normalización de pico a 0.9 como en final.py
    full = full / (np.max(np.abs(full)) + 1e-9) * 0.9
    return full, ground_truth
