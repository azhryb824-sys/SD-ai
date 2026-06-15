import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
OUTPUT_DIR = ROOT / "voice_samples" / "generated"
MANIFEST_PATH = ROOT / "voice_samples" / "all_references.json"
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("TTS_HOME", str(CACHE_DIR / "tts"))
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import numpy as np
import soundfile as sf
import torch
import torchaudio
from TTS.api import TTS


SEGMENTS = [
    (
        "خلونا نتكلم شوية عن الأرقام. واحد، اتنين، تلاتة، أربعة، خمسة، "
        "ستة، سبعة، تمانية، تسعة، عشرة. أحدعش، اتناشر، تلاتاشر، أربعة عشر، "
        "خمستاشر، ستاشر، سبعتاشر، تمنطاشر، تسعتاشر، عشرين. مية، ميتين، "
        "خمسمية، ألف، خمسة آلاف، عشرة آلاف.",
        1.02,
    ),
    (
        "هسع حأتكلم بسرعة شوية، وبعدها حأرجع للكلام الهادي. أحياناً الإنسان "
        "يكون مستعجل عشان عندو موعد مهم،",
        1.13,
    ),
    (
        "وأحياناً يكون مرتاح وما عندو أي ارتباطات، فطريقة الكلام بتختلف "
        "من موقف لموقف.",
        0.96,
    ),
]


def load_wav_without_torchcodec(path, *args, **kwargs):
    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    return torch.from_numpy(audio.T.copy()), sample_rate


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    references = [ROOT / item for item in manifest["references"]]
    missing = [path for path in references if not path.exists()]
    if missing:
        raise SystemExit(f"Missing reference files: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sudanese_numbers_and_pacing.wav"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torchaudio.load = load_wav_without_torchcodec
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    generated = []
    sample_rate = 24000
    silence = np.zeros(int(sample_rate * 0.22), dtype=np.float32)
    for index, (text, speed) in enumerate(SEGMENTS, start=1):
        print(f"Generating segment {index}/{len(SEGMENTS)} at speed {speed}...")
        audio = tts.tts(
            text=text,
            speaker_wav=[str(path) for path in references],
            language="ar",
            split_sentences=True,
            speed=speed,
        )
        generated.append(np.asarray(audio, dtype=np.float32))
        if index < len(SEGMENTS):
            generated.append(silence)

    combined = np.concatenate(generated)
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0.98:
        combined *= 0.98 / peak
    sf.write(str(output_path), combined, sample_rate, subtype="PCM_16")
    print(output_path)
    print(f"Duration: {len(combined) / sample_rate:.2f}s")
    print(f"References used: {len(references)}")


if __name__ == "__main__":
    main()
