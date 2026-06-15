import json
import math
import subprocess
import wave
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "wav"
REPORT_PATH = Path(__file__).resolve().parent / "audio_report.json"
SELECTION_PATH = Path(__file__).resolve().parent / "selected_references.json"
SUPPORTED_EXTENSIONS = {".aac", ".m4a", ".mp3", ".wav", ".flac", ".ogg"}


def audio_metrics(output):
    with wave.open(str(output), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        duration = wav_file.getnframes() / sample_rate

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms_db = 20 * math.log10(max(rms, 1e-8))
    silence_ratio = float(np.mean(np.abs(samples) < 0.004))
    clipping_ratio = float(np.mean(np.abs(samples) >= 0.99))

    duration_score = max(0.0, 1.0 - abs(duration - 11.0) / 18.0)
    level_score = max(0.0, 1.0 - abs(rms_db + 20.0) / 22.0)
    quality_score = (
        duration_score * 35
        + level_score * 35
        + (1.0 - min(silence_ratio, 0.75)) * 25
        + (1.0 - min(clipping_ratio * 100, 1.0)) * 5
    )
    if duration < 3.0:
        quality_score -= 35
    if duration > 35.0:
        quality_score -= 20

    return {
        "duration_seconds": round(duration, 2),
        "rms_dbfs": round(rms_db, 2),
        "peak": round(peak, 4),
        "silence_ratio": round(silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 6),
        "quality_score": round(quality_score, 2),
    }


def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    sources = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not sources:
        raise SystemExit("No audio files were found in the project root.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    for index, source in enumerate(sources, start=1):
        output = OUTPUT_DIR / f"voice_{index:03d}.wav"
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-af",
            (
                "highpass=f=70,lowpass=f=10000,"
                "silenceremove=start_periods=1:start_duration=0.1:"
                "start_threshold=-45dB,areverse,"
                "silenceremove=start_periods=1:start_duration=0.1:"
                "start_threshold=-45dB,areverse,"
                "loudnorm=I=-20:TP=-2:LRA=11"
            ),
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        metrics = audio_metrics(output)
        report.append(
            {
                "source": source.name,
                "output": output.name,
                "recorded_date": datetime.fromtimestamp(
                    source.stat().st_mtime
                ).date().isoformat(),
                **metrics,
                "size_bytes": output.stat().st_size,
                "ffmpeg_messages": completed.stderr.splitlines()[-2:],
            }
        )
        print(
            f"Prepared {output.name} "
            f"({metrics['duration_seconds']:.2f}s, score {metrics['quality_score']:.1f})"
        )

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    eligible = [
        item
        for item in report
        if 3.0 <= item["duration_seconds"] <= 35.0
        and item["clipping_ratio"] < 0.002
        and item["silence_ratio"] < 0.65
    ]
    newest_date = max(item["recorded_date"] for item in eligible)
    newest = sorted(
        (item for item in eligible if item["recorded_date"] == newest_date),
        key=lambda item: item["quality_score"],
        reverse=True,
    )[:3]
    remaining = sorted(
        (item for item in eligible if item not in newest),
        key=lambda item: item["quality_score"],
        reverse=True,
    )
    selected = newest + remaining[: 6 - len(newest)]
    SELECTION_PATH.write_text(
        json.dumps(
            {
                "references": [item["output"] for item in selected],
                "sources": [item["source"] for item in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    total_duration = sum(item["duration_seconds"] for item in report)
    print(f"Prepared {len(report)} files, total duration: {total_duration:.2f}s")
    print("Selected references:", ", ".join(item["output"] for item in selected))


if __name__ == "__main__":
    main()
