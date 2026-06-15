import os
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
OUTPUT_DIR = ROOT / "voice_samples" / "generated"
REFERENCE_DIR = ROOT / "voice_samples" / "wav"
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("TTS_HOME", str(CACHE_DIR / "tts"))
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import torch
import soundfile as sf
import torchaudio
from TTS.api import TTS
from inference.dialect import to_sudanese_text


def load_wav_without_torchcodec(path, *args, **kwargs):
    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    return torch.from_numpy(audio.T.copy()), sample_rate


def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = (
            "يا مرحب بيك، هسع أنا جاهز أساعدك. كان عندك أي سؤال، "
            "قول لي وأنا برد عليك قدر ما بقدر."
        )
    text = to_sudanese_text(text)

    selection = json.loads(
        (ROOT / "voice_samples" / "selected_references.json").read_text(
            encoding="utf-8"
        )
    )
    references = [
        REFERENCE_DIR / filename for filename in selection["references"]
    ]
    missing = [path for path in references if not path.exists()]
    if missing:
        raise SystemExit(f"Missing reference files: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sudanese_full_voice_sample.wav"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torchaudio.load = load_wav_without_torchcodec
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts.tts_to_file(
        text=text,
        speaker_wav=[str(path) for path in references],
        language="ar",
        file_path=str(output_path),
        split_sentences=True,
    )
    print(output_path)


if __name__ == "__main__":
    main()
