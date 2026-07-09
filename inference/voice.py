import hashlib
import json
import os
import queue
import re
import struct
import subprocess
import threading
import wave
from pathlib import Path

from num2words import num2words

from inference.dialect import to_sudanese_text

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
REFERENCE_DIR = ROOT / "voice_samples" / "wav"
OUTPUT_DIR = ROOT / "voice_samples" / "generated"

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("TTS_HOME", str(CACHE_DIR / "tts"))
os.environ.setdefault("COQUI_TOS_AGREED", "1")

DEFAULT_REFERENCES = (
    "voice_002.wav",
    "voice_004.wav",
    "voice_005.wav",
    "voice_006.wav",
)


def selected_reference_names():
    selection_path = ROOT / "voice_samples" / "selected_references.json"
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        references = tuple(payload.get("references", []))
        if references:
            return references
    except (OSError, ValueError, TypeError):
        pass
    return DEFAULT_REFERENCES


VOICE_REFERENCES = {
    "sudanese": selected_reference_names(),
}

_model = None
_model_lock = threading.Lock()
_generation_lock = threading.Lock()
_conditioning_cache = {}
_warmup_state = {"ready": False, "loading": False, "error": None}


def reload_voice_references():
    new_refs = selected_reference_names()
    VOICE_REFERENCES["sudanese"] = new_refs
    _conditioning_cache.clear()
    return list(new_refs)

ARABIC_TO_ASCII_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬",
    "01234567890123456789.,",
)
MONTH_NAMES = (
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
)
CURRENCIES = {
    "ر.س": ("ريال سعودي", "هللة"),
    "ريال": ("ريال سعودي", "هللة"),
    "ريال سعودي": ("ريال سعودي", "هللة"),
    "sar": ("ريال سعودي", "هللة"),
    "$": ("دولار أمريكي", "سنت"),
    "دولار": ("دولار أمريكي", "سنت"),
    "usd": ("دولار أمريكي", "سنت"),
    "€": ("يورو", "سنت"),
    "يورو": ("يورو", "سنت"),
    "eur": ("يورو", "سنت"),
    "جنيه": ("جنيه", "قرش"),
    "sdg": ("جنيه سوداني", "قرش"),
}


def _number_words(value):
    value = value.replace(",", "")
    number = float(value) if "." in value else int(value)
    if isinstance(number, float) and not number.is_integer():
        whole, fraction = value.split(".", 1)
        fraction = fraction.rstrip("0")
        return (
            f"{num2words(int(whole), lang='ar')} فاصلة "
            f"{num2words(int(fraction), lang='ar')}"
        )
    return num2words(int(number), lang="ar")


def prepare_spoken_text(text):
    spoken = text.translate(ARABIC_TO_ASCII_DIGITS)

    def replace_date(match):
        first, second, third = map(int, match.groups())
        if first > 31:
            year, month, day = first, second, third
        else:
            day, month, year = first, second, third
        if not 1 <= month <= 12 or not 1 <= day <= 31:
            return match.group(0)
        return (
            f"{num2words(day, lang='ar', to='ordinal')} من "
            f"{MONTH_NAMES[month - 1]} عام {num2words(year, lang='ar')}"
        )

    spoken = re.sub(
        r"\b(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})\b",
        replace_date,
        spoken,
    )

    currency_pattern = "|".join(
        sorted((re.escape(key) for key in CURRENCIES), key=len, reverse=True)
    )

    def replace_currency(match):
        amount = match.group("amount").replace(",", "")
        currency_key = match.group("currency").lower()
        major, minor = CURRENCIES[currency_key]
        whole, dot, fraction = amount.partition(".")
        result = f"{num2words(int(whole), lang='ar')} {major}"
        if dot and int(fraction or "0"):
            minor_value = int((fraction + "00")[:2])
            result += f" و{num2words(minor_value, lang='ar')} {minor}"
        return result

    spoken = re.sub(
        rf"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*"
        rf"(?P<currency>{currency_pattern})",
        replace_currency,
        spoken,
        flags=re.IGNORECASE,
    )

    spoken = re.sub(
        r"(\d[\d,]*(?:\.\d+)?)\s*%",
        lambda match: f"{_number_words(match.group(1))} بالمائة",
        spoken,
    )
    spoken = re.sub(
        r"\b\d[\d,]*(?:\.\d+)?\b",
        lambda match: _number_words(match.group(0)),
        spoken,
    )
    return re.sub(r"\s+", " ", spoken).strip()


def _clean_speech_text(text, style, limit):
    clean_text = " ".join(text.split()).strip()
    if style == "sudanese":
        clean_text = to_sudanese_text(clean_text)
    return prepare_spoken_text(clean_text)[:limit]


def _output_path(clean_text, style):
    reference_signature = ",".join(VOICE_REFERENCES[style])
    cache_value = f"v6:{style}:{reference_signature}:{clean_text}"
    digest = hashlib.sha256(cache_value.encode("utf-8")).hexdigest()[:20]
    return OUTPUT_DIR / f"reply_{style}_{digest}.wav"


def is_speech_cached(text, style="sudanese"):
    if style not in VOICE_REFERENCES:
        return False
    clean_text = _clean_speech_text(text, style, 700)
    return bool(clean_text) and _output_path(clean_text, style).exists()


def _load_model():
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        import soundfile as sf
        import torch
        import torchaudio
        from TTS.api import TTS

        def load_wav_without_torchcodec(path, *args, **kwargs):
            audio, sample_rate = sf.read(
                str(path),
                dtype="float32",
                always_2d=True,
            )
            return torch.from_numpy(audio.T.copy()), sample_rate

        torchaudio.load = load_wav_without_torchcodec
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        ).to(device)
        return _model


def _references_for(style):
    references = [
        REFERENCE_DIR / filename for filename in VOICE_REFERENCES[style]
    ]
    missing = [path for path in references if not path.exists()]
    if missing:
        raise FileNotFoundError("Voice reference files are missing.")
    return references


def _conditioning_for(model, style):
    cached = _conditioning_cache.get(style)
    if cached is not None:
        return cached

    references = _references_for(style)
    tts_model = model.synthesizer.tts_model
    conditioning = tts_model.get_conditioning_latents(
        audio_path=[str(path) for path in references],
        max_ref_length=10,
        gpt_cond_len=5,
        gpt_cond_chunk_len=5,
    )
    _conditioning_cache[style] = conditioning
    return conditioning


def warm_voice_engine():
    if _warmup_state["ready"] or _warmup_state["loading"]:
        return
    _warmup_state.update(loading=True, error=None)
    try:
        model = _load_model()
        with _generation_lock:
            for style in VOICE_REFERENCES:
                _conditioning_for(model, style)
        _warmup_state["ready"] = True
    except Exception as error:
        _warmup_state["error"] = str(error)
    finally:
        _warmup_state["loading"] = False


def voice_engine_status():
    return dict(_warmup_state)


def _speech_segments(text, max_chars=85):
    sentences = re.split(r"(?<=[.!؟])\s+|\n+", text)
    segments = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            split_at = sentence.rfind(" ", 0, max_chars)
            split_at = split_at if split_at > 40 else max_chars
            chunk, sentence = sentence[:split_at], sentence[split_at:]
            if current:
                segments.append(current)
                current = ""
            segments.append(chunk.strip())
            sentence = sentence.strip()
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                segments.append(current)
            current = sentence
    if current:
        segments.append(current)
    return segments


def _streaming_wav_header(sample_rate=24000):
    data_size = 0x7FFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", data_size + 36)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data_size)
    )


def stream_speech(text, style="sudanese"):
    if style not in VOICE_REFERENCES:
        raise ValueError("Unsupported voice style.")
    clean_text = _clean_speech_text(text, style, 900)
    if not clean_text:
        raise ValueError("Text is required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(clean_text[:700], style)
    if len(clean_text) <= 700 and output_path.exists():
        with output_path.open("rb") as cached_audio:
            while chunk := cached_audio.read(64 * 1024):
                yield chunk
        return

    yield _streaming_wav_header()
    model = _load_model()
    pcm_audio = bytearray()
    with _generation_lock:
        gpt_cond_latent, speaker_embedding = _conditioning_for(model, style)
        for segment in _speech_segments(clean_text):
            chunks = model.synthesizer.tts_model.inference_stream(
                text=segment,
                language="ar",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                stream_chunk_size=2,
                do_sample=False,
                speed=1.1,
                enable_text_splitting=False,
            )
            for chunk in chunks:
                audio = chunk.detach().cpu().numpy()
                audio = (audio.clip(-1, 1) * 32767).astype("<i2")
                audio_bytes = audio.tobytes()
                pcm_audio.extend(audio_bytes)
                yield audio_bytes
    if pcm_audio and len(clean_text) <= 700:
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(pcm_audio)


def stream_speech_mp3(text, style="sudanese"):
    import imageio_ffmpeg

    if style not in VOICE_REFERENCES:
        raise ValueError("Unsupported voice style.")
    clean_text = _clean_speech_text(text, style, 360)
    if not clean_text:
        raise ValueError("Text is required.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output_path = _output_path(clean_text, style)
    if output_path.exists():
        cached_process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(output_path),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "64k",
                "-write_xing",
                "0",
                "-f",
                "mp3",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        while chunk := cached_process.stdout.read(16 * 1024):
            yield chunk
        return

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-write_xing",
        "0",
        "-f",
        "mp3",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    output_queue = queue.Queue(maxsize=12)
    sentinel = object()

    def read_encoded_audio():
        try:
            while chunk := process.stdout.read(4096):
                output_queue.put(chunk)
        finally:
            output_queue.put(sentinel)

    def generate_audio():
        try:
            model = _load_model()
            with _generation_lock:
                gpt_cond_latent, speaker_embedding = _conditioning_for(
                    model, style
                )
                for segment in _speech_segments(clean_text, max_chars=65):
                    chunks = model.synthesizer.tts_model.inference_stream(
                        text=segment,
                        language="ar",
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        stream_chunk_size=2,
                        do_sample=False,
                        speed=1.12,
                        enable_text_splitting=False,
                    )
                    for chunk in chunks:
                        audio = chunk.detach().cpu().numpy()
                        audio_bytes = (
                            audio.clip(-1, 1) * 32767
                        ).astype("<i2").tobytes()
                        process.stdin.write(audio_bytes)
                        process.stdin.flush()
        finally:
            if process.stdin:
                process.stdin.close()

    threading.Thread(
        target=read_encoded_audio,
        name="mp3-stream-reader",
        daemon=True,
    ).start()
    threading.Thread(
        target=generate_audio,
        name="mp3-stream-generator",
        daemon=True,
    ).start()

    while True:
        chunk = output_queue.get()
        if chunk is sentinel:
            break
        yield chunk


def synthesize(text, style="sudanese", force=False):
    if style not in VOICE_REFERENCES:
        raise ValueError("Unsupported voice style.")
    clean_text = _clean_speech_text(text, style, 700)
    if not clean_text:
        raise ValueError("Text is required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(clean_text, style)
    if output_path.exists() and not force:
        return output_path

    model = _load_model()
    with _generation_lock:
        if force or not output_path.exists():
            import soundfile as sf

            gpt_cond_latent, speaker_embedding = _conditioning_for(
                model, style
            )
            result = model.synthesizer.tts_model.inference(
                text=clean_text,
                language="ar",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                do_sample=False,
                speed=1.05,
                enable_text_splitting=False,
            )
            sf.write(output_path, result["wav"], 24000)
    return output_path
