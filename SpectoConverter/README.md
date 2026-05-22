# CTF Spectrogram Flag Embedder

Embeds flag text or an image into the spectrogram of a WAV file. Participants read the flag by opening the audio in spectrogram view.

## Requirements

- Python 3.10+

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Windows: double-click `baslat.bat`.

## Usage

1. Choose **Text** or **Image** mode.
2. Adjust size, font, duration, and frequency range.
3. **Preview spectrogram** — check the source image and spectrogram.
4. **Generate and save WAV** — export the challenge audio file.
5. **Save source PNG** — export the embedded source image (optional).

## Project structure

| File | Description |
|------|-------------|
| `app.py` | GUI (CustomTkinter) |
| `spectro_core.py` | Image → audio, spectrogram preview |
| `requirements.txt` | Dependencies |
| `baslat.bat` | Windows launcher |

## Solving (participants)

- [Audacity](https://www.audacityteam.org/) — Spectrogram view
- [Sonic Visualiser](https://www.sonicvisualiser.org/) — Spectrogram layer

## How it works

Each white pixel in the image produces a short sine tone at a specific time and frequency. The spectrogram displays that energy as readable text.
