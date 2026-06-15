import json
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WAV_DIR = ROOT / "voice_samples" / "wav"
OUTPUT_DIR = ROOT / "voice_samples" / "all_references"
MANIFEST_PATH = ROOT / "voice_samples" / "all_references.json"
SAMPLE_RATE = 24000
MAX_CLIP_SECONDS = 6.0
ANALYSIS_WINDOW_SECONDS = 0.5


def read_mono_wav(path):
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError(f"{path.name} is not mono")
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"{path.name} is not 16-bit PCM")
        if wav_file.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path.name} is not {SAMPLE_RATE} Hz")
        frames = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(frames, dtype="<i2").copy()


def strongest_clip(samples):
    max_samples = int(MAX_CLIP_SECONDS * SAMPLE_RATE)
    if len(samples) <= max_samples:
        return samples

    window_size = int(ANALYSIS_WINDOW_SECONDS * SAMPLE_RATE)
    squared = samples.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    energy = cumulative[window_size:] - cumulative[:-window_size]
    peak_center = int(np.argmax(energy)) + window_size // 2
    start = max(0, min(peak_center - max_samples // 2, len(samples) - max_samples))
    return samples[start : start + max_samples]


def normalize(samples):
    samples = samples.astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0:
        samples *= min(1.0, 29490.0 / peak)
    return np.clip(samples, -32768, 32767).astype("<i2")


def write_wav(path, samples):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


def main():
    sources = sorted(WAV_DIR.glob("voice_*.wav"))
    if not sources:
        raise SystemExit("No prepared WAV references were found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUTPUT_DIR.glob("reference_*.wav"):
        old_file.unlink()

    manifest = []
    for index, source in enumerate(sources, start=1):
        samples = read_mono_wav(source)
        clip = normalize(strongest_clip(samples))
        output = OUTPUT_DIR / f"reference_{index:03d}.wav"
        write_wav(output, clip)
        duration = len(clip) / SAMPLE_RATE
        manifest.append(
            {
                "source": source.name,
                "reference": str(output.relative_to(ROOT)),
                "contributed_seconds": round(duration, 3),
            }
        )
        print(f"{source.name} -> {output.name} ({duration:.2f}s)")

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "references": [item["reference"] for item in manifest],
                "source_count": len(manifest),
                "total_contributed_seconds": round(
                    sum(item["contributed_seconds"] for item in manifest), 3
                ),
                "files": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Built {len(manifest)} references from all prepared recordings.")
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
