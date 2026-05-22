from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ProgressCallback = Callable[[float, str], None]

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class SpectroConfig:
    width: int = 900
    height: int = 450
    font_size: int = 48
    duration_sec: float = 12.0
    sample_rate: int = 44100
    freq_min: int = 400
    freq_max: int = 9000
    amplitude: float = 0.28
    invert: bool = False
    n_fft: int = 2048
    binarize_threshold: int = 230


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/System/Library/Fonts/Menlo.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def binarize_image(img: Image.Image, cfg: SpectroConfig) -> Image.Image:
    gray = img.convert("L")
    if cfg.invert:
        gray = Image.eval(gray, lambda p: 255 - p)
    return gray.point(lambda p: 255 if p >= cfg.binarize_threshold else 0, mode="L")


def prepare_image(img: Image.Image, cfg: SpectroConfig) -> Image.Image:
    return binarize_image(img, cfg)


def text_to_image(text: str, cfg: SpectroConfig) -> Image.Image:
    bg = 255 if cfg.invert else 0
    fg = 0 if cfg.invert else 255
    img = Image.new("L", (cfg.width, cfg.height), color=bg)
    draw = ImageDraw.Draw(img)
    font = _load_font(cfg.font_size)

    lines = text.replace("\r", "").split("\n")
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    spacing = max(4, cfg.font_size // 8)
    total_h = sum(line_heights) + spacing * (len(lines) - 1)
    y = (cfg.height - total_h) // 2

    for i, line in enumerate(lines):
        w = line_widths[i]
        x = (cfg.width - w) // 2
        draw.text((x, y), line, fill=fg, font=font)
        y += line_heights[i] + spacing

    return prepare_image(img, cfg)


def load_image_from_path(path: str | Path, cfg: SpectroConfig) -> Image.Image:
    img = Image.open(path).convert("L")
    return img.resize((cfg.width, cfg.height), Image.Resampling.LANCZOS)


def _snap_freq(freq_hz: float, sample_rate: int, n_fft: int) -> float:
    bin_w = sample_rate / n_fft
    idx = int(round(freq_hz / bin_w))
    idx = max(1, min(idx, n_fft // 2 - 1))
    return idx * bin_w


def image_to_audio(
    img: Image.Image,
    cfg: SpectroConfig,
    on_progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, int]:
    binary = prepare_image(img, cfg)
    pixels = np.array(binary, dtype=np.uint8)
    h, w = pixels.shape

    active = pixels >= 128
    if not np.any(active):
        n_samples = int(cfg.sample_rate * cfg.duration_sec)
        return np.zeros(n_samples, dtype=np.float32), cfg.sample_rate

    ys, xs = np.nonzero(active)
    y_min, y_max = int(ys.min()), int(ys.max())
    y_span = max(y_max - y_min, 1)

    f_min, f_max = float(cfg.freq_min), float(cfg.freq_max)
    n_samples = int(cfg.sample_rate * cfg.duration_sec)
    audio = np.zeros(n_samples, dtype=np.float64)
    samples_per_px = n_samples / w
    n_pixels = len(ys)

    if on_progress:
        on_progress(0.05, "Processing pixels…")

    for i, (y, x) in enumerate(zip(ys, xs)):
        if on_progress and (i % 1500 == 0 or i == n_pixels - 1):
            pct = int(100 * (i + 1) / max(n_pixels, 1))
            on_progress(0.05 + 0.85 * (i + 1) / max(n_pixels, 1), f"Generating audio… {pct}%")
        t_norm = (y - y_min) / y_span
        freq = _snap_freq(
            f_max - t_norm * (f_max - f_min),
            cfg.sample_rate,
            cfg.n_fft,
        )

        center = int((x + 0.5) * samples_per_px)
        half = max(2, int(samples_per_px * 0.85))
        start = max(0, center - half)
        end = min(n_samples, center + half)
        length = end - start
        if length <= 0:
            continue

        t = np.arange(length, dtype=np.float64) / cfg.sample_rate
        window = np.hanning(length)
        tone = cfg.amplitude * window * np.sin(2.0 * np.pi * freq * t)
        audio[start:end] += tone

    if on_progress:
        on_progress(0.95, "Normalizing audio…")

    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak * 0.92

    if on_progress:
        on_progress(1.0, "Done")

    return audio.astype(np.float32), cfg.sample_rate


def render_spectrogram_preview(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop: int | None = None,
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scipy import signal

    if hop is None:
        hop = n_fft // 4

    f, t, Sxx = signal.spectrogram(
        audio,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        mode="magnitude",
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram preview")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def save_wav(audio: np.ndarray, sample_rate: int, path: str | Path) -> None:
    from scipy.io import wavfile

    wavfile.write(str(path), sample_rate, audio)
