import base64
import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference.chat import generate, shorten_answer
from inference.learning import record_interaction, submit_feedback
from inference.voice import (
    is_speech_cached,
    stream_speech,
    stream_speech_mp3,
    synthesize,
    voice_engine_status,
    warm_voice_engine,
)

app = FastAPI(title="المساعد السوداني")
CLOUD_DEPLOYMENT = bool(os.getenv("RENDER"))

KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "540"))  # 9 minutes

def _keepalive_loop():
    ext_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
    port = os.getenv("PORT", "10000")
    while True:
        time.sleep(KEEPALIVE_INTERVAL)
        try:
            if ext_url:
                urllib.request.urlopen(f"{ext_url}/health", timeout=15)
            else:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10)
        except Exception:
            pass

keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True)
keepalive_thread.start()
VOICE_SAMPLE_PATH = (
    Path(__file__).resolve().parent
    / "voice_samples"
    / "generated"
    / "sudanese_full_voice_sample.wav"
)
INSTANT_SPEECH_DIR = (
    Path(__file__).resolve().parent / ".cache" / "instant_speech"
)
EDGE_TTS_PATH = (
    Path(__file__).resolve().parent / "venv" / "Scripts" / "edge-tts.exe"
)
FAST_SPEECH_LOCK = threading.Lock()


@app.on_event("startup")
def start_voice_warmup():
    if CLOUD_DEPLOYMENT:
        return
    threading.Thread(
        target=warm_voice_engine,
        name="voice-engine-warmup",
        daemon=True,
    ).start()


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)
    learning_consent: bool = False
    session_id: str = ""
    response_mode: str = "brief"
    direct_mode: bool = False


class FeedbackRequest(BaseModel):
    interaction_id: str
    helpful: bool


class SpeechRequest(BaseModel):
    text: str
    style: str = "sudanese"


@app.get("/", response_class=HTMLResponse)
def home():
    page = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>المساعد السوداني</title>
  <style>
    :root {
      --ink: #282019;
      --muted: #796e63;
      --paper: #fffaf0;
      --paper-deep: #f7eddc;
      --green: #126b55;
      --green-dark: #084b3b;
      --mint: #dceee5;
      --red: #a94534;
      --gold: #c98b2e;
      --blue: #287d91;
      --line: #e6d8c4;
      --shadow: 0 24px 70px rgba(75, 50, 27, .14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, Arial, sans-serif;
      background:
        radial-gradient(circle at 8% 8%, rgba(40, 125, 145, .13) 0, transparent 27%),
        radial-gradient(circle at 92% 88%, rgba(201, 139, 46, .18) 0, transparent 25%),
        linear-gradient(135deg, #f7efe2 0%, #fdf8ee 48%, #f3e8d6 100%);
      overflow: hidden;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .22;
      background-image:
        linear-gradient(30deg, transparent 47%, rgba(169, 69, 52, .12) 48% 52%, transparent 53%),
        linear-gradient(-30deg, transparent 47%, rgba(18, 107, 85, .10) 48% 52%, transparent 53%);
      background-size: 42px 72px;
      mask-image: linear-gradient(to bottom, black, transparent 32%);
    }
    .page {
      position: relative;
      width: min(1240px, 95vw);
      min-height: 100vh;
      margin: auto;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      gap: 20px;
      padding: 22px 0;
    }
    aside, main {
      border: 1px solid rgba(111, 76, 41, .13);
      border-radius: 30px;
      background: rgba(255, 250, 240, .91);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    aside {
      position: relative;
      padding: 32px 27px 26px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    aside::before {
      content: "";
      position: absolute;
      inset: 0 0 auto;
      height: 9px;
      background: linear-gradient(90deg,
        var(--red) 0 20%, var(--gold) 20% 40%,
        var(--green) 40% 60%, var(--blue) 60% 80%, var(--red) 80%);
    }
    aside::after {
      content: "";
      position: absolute;
      width: 190px;
      height: 190px;
      left: -92px;
      bottom: -72px;
      border: 24px double rgba(40, 125, 145, .08);
      transform: rotate(45deg);
    }
    .mark {
      width: 58px;
      height: 58px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(169, 69, 52, .25);
      border-radius: 50% 50% 45% 45%;
      color: var(--paper);
      background:
        linear-gradient(145deg, var(--green), var(--green-dark));
      box-shadow: inset 0 0 0 5px rgba(255,255,255,.12), 0 9px 22px rgba(18,107,85,.2);
      font-size: 24px;
      font-weight: 800;
    }
    .brand-kicker {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 20px;
      color: var(--red);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .2px;
    }
    .brand-kicker::before {
      content: "";
      width: 28px;
      height: 4px;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--red), var(--gold), var(--green));
    }
    aside h1 { margin: 8px 0 8px; font-size: 27px; letter-spacing: -.3px; }
    aside p { margin: 0; color: var(--muted); line-height: 1.8; }
    .status {
      display: flex;
      align-items: center;
      gap: 9px;
      width: fit-content;
      margin-top: 22px;
      padding: 8px 11px;
      border: 1px solid rgba(18,107,85,.15);
      border-radius: 999px;
      color: var(--green);
      background: rgba(220,238,229,.55);
      font-size: 14px;
      font-weight: 650;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #35b77e;
      box-shadow: 0 0 0 5px rgba(53, 183, 126, .13);
    }
    .tips { position: relative; z-index: 1; margin-top: auto; padding-top: 32px; }
    .tips span {
      display: block;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .4px;
    }
    .suggestion {
      width: 100%;
      margin-top: 9px;
      padding: 13px 14px;
      border: 1px solid var(--line);
      border-radius: 15px;
      color: var(--ink);
      background: rgba(255,255,255,.48);
      text-align: right;
      font: inherit;
      cursor: pointer;
      transition: .2s ease;
    }
    .suggestion:hover {
      border-color: rgba(18,107,85,.35);
      background: var(--mint);
      transform: translateX(-3px);
    }
    main {
      position: relative;
      min-height: calc(100vh - 44px);
      height: calc(100vh - 44px);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    main::before {
      content: "";
      position: absolute;
      inset: 0 0 auto;
      height: 7px;
      z-index: 5;
      background:
        repeating-linear-gradient(90deg,
          var(--green) 0 28px, var(--gold) 28px 42px,
          var(--red) 42px 70px, var(--blue) 70px 84px);
    }
    .topbar {
      min-height: 94px;
      padding: 25px 28px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      background: rgba(255,250,240,.72);
    }
    .topbar strong { font-size: 19px; }
    .topbar small { display: block; margin-top: 5px; color: var(--muted); }
    .conversation-title {
      display: flex;
      align-items: center;
      gap: 11px;
      min-width: max-content;
    }
    .mobile-mark {
      display: none;
      width: 42px;
      height: 42px;
      place-items: center;
      border-radius: 15px;
      color: white;
      background: linear-gradient(145deg, var(--green), var(--green-dark));
      box-shadow: inset 0 0 0 3px rgba(255,255,255,.11);
      font-weight: 800;
    }
    .topbar-controls {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .voice-test {
      flex: 0 0 auto;
      border: 1px solid rgba(18,107,85,.23);
      border-radius: 999px;
      padding: 9px 14px;
      color: var(--green);
      background: rgba(220,238,229,.72);
      font: 14px inherit;
      cursor: pointer;
      transition: .2s ease;
    }
    .voice-test:hover { transform: translateY(-1px); background: var(--mint); }
    .voice-test:disabled { opacity: .55; cursor: wait; }
    .response-mode {
      border: 1px solid rgba(201,139,46,.35);
      border-radius: 999px;
      padding: 8px 10px;
      color: #72501f;
      background: #fff8e9;
      font: 13px inherit;
      cursor: pointer;
    }
    .direct-toggle.active {
      color: white;
      border-color: var(--green);
      background: var(--green);
    }
    .direct-status {
      min-height: 22px;
      padding: 0 7% 8px;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
    }
    .direct-status.listening { color: #b33a3a; }
    .direct-status.speaking { color: var(--green); }
    .learning-consent {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      margin-inline-start: 10px;
      color: var(--muted);
      font-size: 12px;
      cursor: pointer;
    }
    .learning-consent input { width: 16px; height: 16px; accent-color: var(--green); }
    .cloud-deployment #direct-toggle,
    .cloud-deployment #voice-test-sudanese,
    .cloud-deployment #microphone,
    .cloud-deployment .welcome-tags span:last-child {
      display: none;
    }
    #chat {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 34px 7% 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      scrollbar-color: rgba(18,107,85,.3) transparent;
    }
    .welcome {
      margin: auto;
      max-width: 560px;
      text-align: center;
    }
    .nile-line {
      width: 122px;
      height: 15px;
      margin: 0 auto 24px;
      border-top: 3px solid var(--blue);
      border-radius: 50%;
      opacity: .75;
      transform: rotate(-3deg);
    }
    .nile-line::before,
    .nile-line::after {
      content: "";
      display: block;
      width: 74px;
      height: 11px;
      margin: 4px auto 0;
      border-top: 2px solid rgba(40,125,145,.48);
      border-radius: 50%;
    }
    .nile-line::after {
      width: 42px;
      margin-top: 2px;
      opacity: .65;
    }
    .welcome .symbol {
      position: relative;
      width: 82px;
      height: 82px;
      margin: 0 auto 24px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(201,139,46,.32);
      border-radius: 28px 28px 42px 42px;
      color: #fff8e9;
      background: linear-gradient(145deg, var(--green), var(--green-dark));
      box-shadow: 0 15px 36px rgba(18,107,85,.2);
      font-size: 29px;
      transform: rotate(45deg);
    }
    .welcome .symbol::before {
      content: "";
      position: absolute;
      inset: 8px;
      border: 1px solid rgba(255,255,255,.25);
      border-radius: 20px 20px 32px 32px;
    }
    .welcome .symbol span { transform: rotate(-45deg); }
    .welcome h2 { margin: 0 0 10px; font-size: clamp(27px, 4vw, 40px); }
    .welcome p { margin: 0; color: var(--muted); line-height: 1.8; }
    .welcome-tags {
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 20px;
    }
    .welcome-tags span {
      padding: 7px 11px;
      border: 1px solid rgba(18,107,85,.15);
      border-radius: 999px;
      color: var(--green);
      background: rgba(220,238,229,.48);
      font-size: 12px;
      font-weight: 700;
    }
    .message {
      max-width: min(78%, 580px);
      padding: 14px 18px;
      border-radius: 20px;
      line-height: 1.8;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      animation: rise .22s ease;
    }
    .message-tools {
      display: flex;
      justify-content: flex-start;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 9px;
    }
    .speak {
      border: 0;
      border-radius: 10px;
      padding: 6px 10px;
      color: var(--green);
      background: var(--mint);
      font: 13px inherit;
      cursor: pointer;
    }
    .speak:disabled { opacity: .55; cursor: wait; }
    .feedback {
      border: 1px solid #cfe3da;
      border-radius: 10px;
      padding: 5px 9px;
      color: var(--green);
      background: white;
      font: 13px inherit;
      cursor: pointer;
    }
    .feedback.selected { color: white; border-color: var(--green); background: var(--green); }
    .feedback:disabled { cursor: default; opacity: .75; }
    .message a {
      color: var(--green);
      font-weight: 650;
      text-decoration: underline;
      text-underline-offset: 3px;
    }
    .user a { color: white; }
    .bot {
      align-self: flex-start;
      border: 1px solid var(--line);
      border-bottom-right-radius: 5px;
      background: rgba(255,255,255,.86);
      box-shadow: 0 8px 22px rgba(75,50,27,.06);
    }
    .user {
      align-self: flex-end;
      border-bottom-left-radius: 5px;
      color: white;
      background: linear-gradient(145deg, var(--green), var(--green-dark));
      box-shadow: 0 9px 24px rgba(18,107,85,.17);
    }
    .typing { color: var(--muted); }
    form {
      flex: 0 0 auto;
      margin: 0 6% 22px;
      display: flex;
      gap: 10px;
      padding: 8px;
      border: 1px solid rgba(111,76,41,.16);
      border-radius: 22px;
      background: rgba(255,255,255,.92);
      box-shadow: 0 14px 38px rgba(75,50,27,.1);
      z-index: 2;
    }
    input {
      min-width: 0;
      flex: 1;
      border: 0;
      padding: 13px 15px;
      color: var(--ink);
      background: transparent;
      font: inherit;
      outline: none;
    }
    #send {
      width: 50px;
      height: 50px;
      border: 0;
      border-radius: 17px;
      color: white;
      background: linear-gradient(145deg, var(--green), var(--green-dark));
      font-size: 20px;
      cursor: pointer;
      box-shadow: 0 7px 16px rgba(18,107,85,.2);
    }
    #send:hover { background: var(--green-dark); }
    #send:disabled { opacity: .55; cursor: wait; }
    #microphone {
      width: 50px;
      height: 50px;
      flex: 0 0 auto;
      border: 1px solid rgba(40,125,145,.23);
      border-radius: 17px;
      color: var(--green);
      background: #e2f0ed;
      font-size: 20px;
      cursor: pointer;
    }
    #microphone.listening {
      color: white;
      border-color: #b33a3a;
      background: #b33a3a;
      animation: pulse 1.25s infinite;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(7px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
      50% { box-shadow: 0 0 0 7px rgba(179, 58, 58, .14); }
    }
    @media (max-width: 760px) {
      .page { display: block; width: 100%; padding: 0; }
      aside { display: none; }
      main { min-height: 100dvh; height: 100dvh; border: 0; border-radius: 0; }
      .topbar {
        min-height: auto;
        padding: 20px 16px 13px;
        align-items: flex-start;
        flex-direction: column;
        gap: 12px;
      }
      .conversation-title { min-width: 0; }
      .mobile-mark { display: grid; flex: 0 0 auto; }
      .topbar strong { font-size: 18px; }
      .topbar small { font-size: 12px; }
      .topbar-controls {
        width: 100%;
        justify-content: flex-start;
        flex-wrap: nowrap;
        overflow-x: auto;
        padding: 0 0 4px;
        scrollbar-width: none;
      }
      .topbar-controls::-webkit-scrollbar { display: none; }
      .voice-test, .response-mode { min-height: 38px; white-space: nowrap; }
      .learning-consent { flex: 0 0 auto; white-space: nowrap; }
      #chat { padding: 24px 16px 18px; }
      .welcome { max-width: 330px; }
      .welcome .symbol { width: 68px; height: 68px; margin-bottom: 21px; }
      .welcome h2 { font-size: 29px; }
      .welcome p { font-size: 14px; }
      .nile-line { margin-bottom: 19px; }
      .welcome-tags { margin-top: 16px; gap: 6px; }
      .welcome-tags span { padding: 6px 9px; font-size: 11px; }
      form { margin: 0 12px max(12px, env(safe-area-inset-bottom)); }
      .message { max-width: 91%; }
      #send, #microphone { width: 48px; height: 48px; }
    }
    @media (max-width: 390px) {
      .topbar { padding-inline: 13px; }
      #chat { padding-inline: 13px; }
      form { margin-inline: 8px; gap: 7px; padding: 7px; }
      input { padding-inline: 9px; }
    }
  </style>
</head>
<body class="__DEPLOYMENT_CLASS__">
  <div class="page">
    <aside>
      <div class="mark">س</div>
      <div class="brand-kicker">ونسة سودانية ذكية</div>
      <h1>مساعدك السوداني</h1>
      <p>اسأل، ابحث واتونس بالدارجي السوداني. بنحلّل ليك الكلام وبنجيب الخلاصة المفيدة.</p>
      <div class="status"><i class="dot"></i> جاهز للونسة والسؤال</div>
      <div class="tips">
        <span>ممكن تبدأ بي واحدة من دي</span>
        <button class="suggestion" type="button">السلام عليكم</button>
        <button class="suggestion" type="button">ورّيني بتقدر تساعدني كيف؟</button>
        <button class="suggestion" type="button">أديني الخلاصة عن السودان</button>
      </div>
    </aside>
    <main>
      <header class="topbar">
        <div class="conversation-title">
          <span class="mobile-mark">س</span>
          <div>
            <strong>ونسة جديدة</strong>
            <small>بالدارجي السوداني • اسأل براحتك</small>
          </div>
        </div>
        <div class="topbar-controls">
          <button class="voice-test direct-toggle" id="direct-toggle" type="button">ونسة صوتية مباشرة</button>
          <button class="voice-test" id="voice-test-sudanese" type="button">جرّب صوتي السوداني</button>
          <select class="response-mode" id="response-mode" aria-label="طول الإجابة">
            <option value="brief">على السريع</option>
            <option value="balanced">شرح وسط</option>
            <option value="detailed">بالتفصيل</option>
          </select>
          <label class="learning-consent" title="تُحفظ الرسائل محليًا بعد إخفاء البيانات الحساسة">
            <input id="learning-consent" type="checkbox">
            خلّي رسائلك تساعدنا نتحسّن
          </label>
        </div>
      </header>
      <section id="chat">
        <div class="welcome" id="welcome">
          <div class="symbol"><span>✦</span></div>
          <div class="nile-line" aria-hidden="true"></div>
          <h2>يا مراحب، البيت بيتك</h2>
          <p>قول الدايرو براحتك. بسأل معاك، بفتّش، وبديك الكلام المفيد من غير لف كتير.</p>
          <div class="welcome-tags" aria-label="قدرات المساعد">
            <span>دارجي سوداني</span>
            <span>بحث وتحليل</span>
            <span>ونسة صوتية</span>
          </div>
        </div>
      </section>
      <div class="direct-status" id="direct-status" aria-live="polite"></div>
      <form id="form">
        <input id="message" autocomplete="off" placeholder="اكتب الداير تقولو هنا..." autofocus>
        <button id="microphone" type="button" aria-label="بدء الاستماع">●</button>
        <button id="send" type="submit" aria-label="إرسال">↑</button>
      </form>
    </main>
  </div>
  <script>
    const form = document.querySelector("#form");
    const input = document.querySelector("#message");
    const chat = document.querySelector("#chat");
    const send = document.querySelector("#send");
    const microphone = document.querySelector("#microphone");
    const directToggle = document.querySelector("#direct-toggle");
    const directStatus = document.querySelector("#direct-status");
    const voiceTestSudanese = document.querySelector("#voice-test-sudanese");
    const responseMode = document.querySelector("#response-mode");
    const learningConsent = document.querySelector("#learning-consent");
    const history = [];
    const speechCache = new Map();
    const sessionId =
      localStorage.getItem("assistant-session-id") || crypto.randomUUID();
    localStorage.setItem("assistant-session-id", sessionId);
    learningConsent.checked =
      localStorage.getItem("assistant-learning-consent") === "true";
    learningConsent.addEventListener("change", () => {
      localStorage.setItem(
        "assistant-learning-consent",
        String(learningConsent.checked)
      );
    });
    responseMode.value =
      localStorage.getItem("assistant-response-mode") || "brief";
    responseMode.addEventListener("change", () => {
      localStorage.setItem("assistant-response-mode", responseMode.value);
    });
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let directMode = false;
    let listening = false;
    let assistantSpeaking = false;
    let activeAudio = null;
    let recognitionRestartTimer = null;
    let transcriptSubmitTimer = null;
    let finalTranscript = "";

    function speechText(text) {
      const cleanText = text
        .split("\\n")
        .filter((line) =>
          !line.trim().startsWith("المصدر:") &&
          !line.trim().startsWith("مرجع البيانات:")
        )
        .join(" ")
        .replace(/https?:\\/\\/[^\\s]+/g, "")
        .replace(/\\s+/g, " ")
        .trim();
      const sentences = cleanText.match(/[^.!؟]+[.!؟]?/g) || [cleanText];
      const selected = [];
      let length = 0;
      for (const sentence of sentences) {
        const cleanSentence = sentence.trim();
        if (!cleanSentence) continue;
        if (selected.length && length + cleanSentence.length > 180) break;
        selected.push(cleanSentence);
        length += cleanSentence.length;
        if (selected.length === 2) break;
      }
      return selected.join(" ").trim();
    }

    function preferredArabicVoice() {
      const voices = window.speechSynthesis?.getVoices() || [];
      return (
        voices.find((voice) => voice.lang.toLowerCase() === "ar-sa") ||
        voices.find((voice) => voice.lang.toLowerCase().startsWith("ar")) ||
        null
      );
    }

    async function playFastSpeech(text) {
      const cleanText = speechText(text);
      if (!cleanText) return false;
      assistantSpeaking = true;
      stopListening();
      const params = new URLSearchParams({text: cleanText});
      const audio = new Audio(`/speech/instant?${params.toString()}`);
      activeAudio = audio;
      audio.preload = "auto";
      audio.load();
      let cached = false;
      try {
        const status = await fetch(`/speech/instant-status?${params.toString()}`);
        cached = (await status.json()).cached;
      } catch {}
      const playReply = () => {
        audio.play().catch(() => {
          assistantSpeaking = false;
          setDirectStatus("اضغط جوة الصفحة عشان تسمح بتشغيل الصوت.");
        });
      };
      if (cached) {
        playReply();
      } else {
        const acknowledgement = new Audio("/speech/ack");
        acknowledgement.addEventListener("ended", playReply);
        acknowledgement.addEventListener("error", playReply);
        acknowledgement.play().catch(playReply);
      }
      audio.addEventListener("playing", () => {
        setDirectStatus("المساعد بتكلم هسع...", "speaking");
      });
      audio.addEventListener("ended", () => {
        activeAudio = null;
        assistantSpeaking = false;
        if (directMode) {
          setDirectStatus("اتكلم هسع...", "listening");
          scheduleListening(250);
        } else {
          setDirectStatus();
        }
      });
      audio.addEventListener("error", () => {
        activeAudio = null;
        assistantSpeaking = false;
        setDirectStatus("ما قدرنا نشغل الصوت. جرّب تاني.");
        if (directMode) {
          scheduleListening(500);
        }
      });
      return true;
    }

    function setDirectStatus(text = "", state = "") {
      directStatus.textContent = text;
      directStatus.className = `direct-status ${state}`;
    }

    function stopListening() {
      clearTimeout(recognitionRestartTimer);
      recognitionRestartTimer = null;
      if (recognition && listening) {
        try {
          recognition.abort();
        } catch {}
      }
      listening = false;
      microphone.classList.remove("listening");
      microphone.setAttribute("aria-label", "بدء الاستماع");
    }

    function scheduleListening(delay = 300) {
      clearTimeout(recognitionRestartTimer);
      if (!directMode || assistantSpeaking || send.disabled) return;
      recognitionRestartTimer = setTimeout(startListening, delay);
    }

    function startListening() {
      if (!recognition || listening || assistantSpeaking || send.disabled) return;
      clearTimeout(recognitionRestartTimer);
      recognitionRestartTimer = null;
      finalTranscript = "";
      input.value = "";
      try {
        recognition.start();
      } catch {
        scheduleListening(400);
      }
    }

    if (SpeechRecognition) {
      recognition = new SpeechRecognition();
      recognition.lang = "ar-SA";
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 3;

      recognition.addEventListener("start", () => {
        listening = true;
        microphone.classList.add("listening");
        microphone.setAttribute("aria-label", "إيقاف الاستماع");
        setDirectStatus("سامعك هسع...", "listening");
      });

      recognition.addEventListener("result", (event) => {
        let interimTranscript = "";
        let receivedFinal = false;
        for (let index = event.resultIndex; index < event.results.length; index++) {
          const part = event.results[index][0].transcript.trim();
          if (event.results[index].isFinal) {
            finalTranscript = `${finalTranscript} ${part}`.trim();
            receivedFinal = true;
          } else {
            interimTranscript += ` ${part}`;
          }
        }
        input.value = `${finalTranscript} ${interimTranscript}`.trim();
        if (receivedFinal && finalTranscript) {
          clearTimeout(transcriptSubmitTimer);
          transcriptSubmitTimer = setTimeout(() => {
            const message = finalTranscript.trim();
            finalTranscript = "";
            stopListening();
            if (message) submitMessage(message);
          }, 650);
        }
      });

      recognition.addEventListener("end", () => {
        listening = false;
        microphone.classList.remove("listening");
        if (!assistantSpeaking && !send.disabled && directMode && !finalTranscript) {
          setDirectStatus("سامعك، اتكلم براحتك...", "listening");
          scheduleListening(350);
        }
      });

      recognition.addEventListener("error", (event) => {
        listening = false;
        microphone.classList.remove("listening");
        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
          directMode = false;
          directToggle.classList.remove("active");
          directToggle.textContent = "دردشة صوتية مباشرة";
          setDirectStatus("لازم تسمح للمتصفح يستخدم المايك.", "");
        } else if (event.error === "no-speech") {
          setDirectStatus("أنا سامعك، اتكلم لمن تكون جاهز...", "listening");
          scheduleListening(350);
        } else if (event.error !== "aborted") {
          setDirectStatus("الصوت ما كان واضح، اتكلم تاني.", "");
          scheduleListening(500);
        }
      });
    } else {
      microphone.disabled = true;
      directToggle.disabled = true;
      setDirectStatus("المتصفح دا ما بدعم التعرّف الصوتي المباشر.");
    }

    async function sendFeedback(interactionId, helpful, buttons) {
      buttons.forEach((button) => {
        button.disabled = true;
      });
      try {
        const response = await fetch("/feedback", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            interaction_id: interactionId,
            helpful
          })
        });
        if (!response.ok) throw new Error();
        const data = await response.json();
        const selected = helpful ? buttons[0] : buttons[1];
        selected.classList.add("selected");
        selected.textContent = helpful
          ? (data.learned ? "تم التعلّم" : "مفيد")
          : "تم التسجيل";
      } catch {
        buttons.forEach((button) => {
          button.disabled = false;
        });
      }
    }

    function addMessage(text, role, extraClass = "", interactionId = null) {
      document.querySelector("#welcome")?.remove();
      const bubble = document.createElement("div");
      bubble.className = `message ${role} ${extraClass}`;
      if (role === "bot") {
        const urlPattern = /(https?:\\/\\/[^\\s]+)/g;
        let lastIndex = 0;
        for (const match of text.matchAll(urlPattern)) {
          bubble.append(document.createTextNode(text.slice(lastIndex, match.index)));
          const link = document.createElement("a");
          link.href = match[0];
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.textContent = "فتح المصدر";
          bubble.appendChild(link);
          lastIndex = match.index + match[0].length;
        }
        bubble.append(document.createTextNode(text.slice(lastIndex)));
      } else {
        bubble.textContent = text;
      }
      if (role === "bot" && !extraClass.includes("typing")) {
        const tools = document.createElement("div");
        tools.className = "message-tools";
        const sudaneseSpeak = document.createElement("button");
        sudaneseSpeak.className = "speak";
        sudaneseSpeak.type = "button";
        sudaneseSpeak.textContent = "تشغيل فوري";
        sudaneseSpeak.addEventListener(
          "click",
          () => playFastSpeech(text)
        );
        tools.appendChild(sudaneseSpeak);
        const clonedSpeak = document.createElement("button");
        clonedSpeak.className = "speak";
        clonedSpeak.type = "button";
        clonedSpeak.textContent = "بصوتك المستنسخ";
        clonedSpeak.title = "جودة أعلى، لكنه أبطأ على المعالج الحالي";
        clonedSpeak.addEventListener(
          "click",
          () => playSpeech(text, clonedSpeak, "sudanese")
        );
        tools.appendChild(clonedSpeak);
        if (interactionId) {
          const helpful = document.createElement("button");
          helpful.className = "feedback";
          helpful.type = "button";
          helpful.textContent = "مفيد";
          const unhelpful = document.createElement("button");
          unhelpful.className = "feedback";
          unhelpful.type = "button";
          unhelpful.textContent = "غير مفيد";
          const feedbackButtons = [helpful, unhelpful];
          helpful.addEventListener(
            "click",
            () => sendFeedback(interactionId, true, feedbackButtons)
          );
          unhelpful.addEventListener(
            "click",
            () => sendFeedback(interactionId, false, feedbackButtons)
          );
          tools.appendChild(helpful);
          tools.appendChild(unhelpful);
        }
        bubble.appendChild(tools);
      }
      chat.appendChild(bubble);
      chat.scrollTop = chat.scrollHeight;
      return bubble;
    }

    async function playSpeech(text, button = null, style = "sudanese") {
      const originalText = button?.textContent || "";
      const cleanText = speechText(text);
      const cacheKey = `${style}:${cleanText}`;
      if (button) {
        button.disabled = true;
        button.textContent = speechCache.has(cacheKey)
          ? "يعمل الآن"
          : "جارٍ تجهيز الصوت...";
      }
      assistantSpeaking = true;
      stopListening();
      if (directMode) {
        try {
          const statusResponse = await fetch("/speech/status");
          const status = await statusResponse.json();
          setDirectStatus(
            status.ready
              ? "بجهز الرد الصوتي..."
              : "محرك الصوت بجهز أول مرة...",
            "speaking"
          );
        } catch {
          setDirectStatus("بجهز الرد الصوتي...", "speaking");
        }
      }
      try {
        const params = new URLSearchParams({text: cleanText, style});
        const audio = new Audio(`/speech/stream-fast?${params.toString()}`);
        activeAudio = audio;
        audio.addEventListener("ended", () => {
          activeAudio = null;
          speechCache.set(cacheKey, true);
          assistantSpeaking = false;
          if (button) {
            button.textContent = originalText;
            button.disabled = false;
          }
          if (directMode) {
            setDirectStatus("اتكلم هسع...", "listening");
            scheduleListening(250);
          } else {
            setDirectStatus();
          }
        });
        await audio.play();
        if (button) button.textContent = "يعمل الآن";
        if (directMode) setDirectStatus("المساعد بتكلم...", "speaking");
      } catch {
        activeAudio = null;
        assistantSpeaking = false;
        if (button) {
          button.textContent = "الصوت ما اشتغل";
          setTimeout(() => {
            button.textContent = originalText;
            button.disabled = false;
          }, 2500);
        }
        if (directMode) {
          setDirectStatus("ما قدرنا نشغل الرد الصوتي. ممكن تواصل كلامك.", "");
          scheduleListening(300);
        }
      }
    }

    async function submitMessage(message) {
      if (!message || send.disabled) return;
      addMessage(message, "user");
      const requestHistory = history.slice(-16);
      input.value = "";
      send.disabled = true;
      stopListening();
      if (directMode) setDirectStatus("بفكر في الإجابة...", "");
      const typing = addMessage("بكتب هسع...", "bot", "typing");
      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            message,
            history: requestHistory,
            learning_consent: learningConsent.checked,
            session_id: sessionId,
            response_mode: responseMode.value,
            direct_mode: directMode
          })
        });
        if (!response.ok) throw new Error();
        const data = await response.json();
        typing.remove();
        addMessage(data.response, "bot", "", data.interaction_id);
        history.push(
          {role: "user", content: message},
          {role: "assistant", content: data.response}
        );
        if (directMode) {
          if (!(await playFastSpeech(data.response))) {
            await playSpeech(data.response, null, "sudanese");
          }
        }
      } catch {
        typing.textContent = "ما قدرنا نتصل بالمساعد. جرّب تاني.";
        if (directMode) {
          setDirectStatus("الاتصال ما تم. اتكلم تاني.", "");
          scheduleListening(500);
        }
      } finally {
        send.disabled = false;
        if (!directMode) input.focus();
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      submitMessage(input.value.trim());
    });

    microphone.addEventListener("click", () => {
      if (!recognition) return;
      if (listening) {
        stopListening();
        setDirectStatus("وقفنا الاستماع.");
      } else {
        startListening();
      }
    });

    directToggle.addEventListener("click", () => {
      directMode = !directMode;
      directToggle.classList.toggle("active", directMode);
      directToggle.textContent = directMode
        ? "وقف الدردشة المباشرة"
        : "دردشة صوتية مباشرة";
      if (directMode) {
        setDirectStatus("تمام، أنا سامعك. اتكلم براحتك...", "listening");
        startListening();
      } else {
        clearTimeout(transcriptSubmitTimer);
        finalTranscript = "";
        stopListening();
        if (activeAudio) {
          activeAudio.pause();
          activeAudio.src = "";
          activeAudio = null;
        }
        assistantSpeaking = false;
        setDirectStatus();
      }
    });

    document.querySelectorAll(".suggestion").forEach((button) => {
      button.addEventListener("click", () => submitMessage(button.textContent.trim()));
    });

    voiceTestSudanese.addEventListener("click", () => {
      voiceTestSudanese.disabled = true;
      voiceTestSudanese.textContent = "الصوت شغال...";
      const audio = new Audio("/speech/sample");
      audio.addEventListener("ended", () => {
        voiceTestSudanese.disabled = false;
        voiceTestSudanese.textContent = "جرّب الصوت السوداني";
      });
      audio.addEventListener("error", () => {
        voiceTestSudanese.disabled = false;
        voiceTestSudanese.textContent = "الصوت ما اشتغل";
      });
      audio.play();
    });
  </script>
</body>
</html>
"""
    return page.replace(
        "__DEPLOYMENT_CLASS__",
        "cloud-deployment" if CLOUD_DEPLOYMENT else "",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "sd-ai",
        "voice_mode": "local" if not CLOUD_DEPLOYMENT else "disabled-on-cloud",
        "keepalive": True,
        "keepalive_interval_s": KEEPALIVE_INTERVAL,
    }


class TrainingRequest(BaseModel):
    audio: str = ""
    name: str = ""


@app.post("/training/voice")
def training_voice(request: TrainingRequest):
    audio_base64 = request.audio.strip()
    name = request.name.strip() or f"sample_{int(time.time())}.wav"
    if not audio_base64:
        return {"ok": False, "error": "Audio data is required."}
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return {"ok": False, "error": f"Invalid base64: {e}"}
    app_root = Path(__file__).resolve().parent
    ref_dir = app_root / "voice_samples" / "wav"
    ref_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in "._-") or f"sample_{int(time.time())}.wav"
    dest = ref_dir / f"user_{safe_name}"
    dest.write_bytes(audio_bytes)
    # Convert webm/mp3/ogg to wav for Coqui TTS compatibility
    wav_name = safe_name.rsplit(".", 1)[0] + ".wav"
    wav_dest = ref_dir / f"user_{wav_name}"
    if dest.suffix.lower() != ".wav" and dest != wav_dest:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(dest), "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", str(wav_dest)],
                check=True, capture_output=True, timeout=30
            )
            dest.unlink(missing_ok=True)
            dest = wav_dest
            safe_name = f"user_{wav_name}"
        except Exception:
            pass
    # Update selected_references.json
    selection_path = app_root / "voice_samples" / "selected_references.json"
    refs = {"references": []}
    try:
        refs = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    new_ref = dest.name
    existing = list(refs.get("references", []))
    if new_ref not in existing:
        existing.insert(0, new_ref)
    refs["references"] = existing[:20]
    selection_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
    # Clear cached audio
    cache_dir = app_root / ".cache"
    for f in cache_dir.rglob("reply_*"):
        try: f.unlink()
        except: pass
    # Reload references in memory + recondition model
    try:
        from inference.voice import reload_voice_references, _model, _model_lock
        reload_voice_references()
        # Clear model speaker conditioning so it picks up new references
        with _model_lock:
            if _model is not None:
                try:
                    _model.synthesizer.tts_model.gpt_cond_latent = None
                    _model.synthesizer.tts_model.speaker_embedding = None
                except Exception:
                    pass
    except Exception:
        pass
    return {"ok": True, "name": new_ref, "totalRefs": len(existing)}


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        return {"response": "اكتب رسالتك وأنا برد عليك."}
    history = [
        item
        for item in request.history[-16:]
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]
    response = generate(
        message,
        history=history,
        response_mode="brief" if request.direct_mode else request.response_mode,
    )
    if request.direct_mode:
        response = shorten_answer(response, max_chars=260)
    interaction_id = record_interaction(
        message,
        response,
        history,
        request.session_id,
        request.learning_consent,
    )
    return {
        "response": response,
        "interaction_id": interaction_id,
        "learning_saved": bool(interaction_id),
    }


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    recorded, learned = submit_feedback(
        request.interaction_id,
        request.helpful,
    )
    if not recorded:
        return {"recorded": False, "learned": False}
    return {"recorded": True, "learned": learned}


@app.post("/speech")
async def speech(request: SpeechRequest):
    text = request.text.strip()
    if not text:
        return {"error": "Text is required."}
    output_path = await run_in_threadpool(synthesize, text, request.style)
    return FileResponse(output_path, media_type="audio/wav", filename="reply.wav")


@app.get("/speech/status")
def speech_status():
    return voice_engine_status()


@app.get("/speech/cache-status")
def speech_cache_status(text: str, style: str = "sudanese"):
    return {"cached": is_speech_cached(text, style)}


def instant_speech_path(text):
    clean_text = " ".join(text.split()).strip()[:360]
    digest = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:20]
    return INSTANT_SPEECH_DIR / f"instant_{digest}.mp3"


def generate_instant_speech(text):
    if CLOUD_DEPLOYMENT:
        raise RuntimeError("Instant speech is available in the local edition.")
    output_path = instant_speech_path(text)
    if output_path.exists():
        return output_path
    INSTANT_SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    with FAST_SPEECH_LOCK:
        if output_path.exists():
            return output_path
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )
        subprocess.run(
            [
                str(EDGE_TTS_PATH),
                "--voice",
                "ar-SA-HamedNeural",
                "--text",
                text[:360],
                "--write-media",
                str(output_path),
            ],
            check=True,
            timeout=45,
            creationflags=creation_flags,
        )
    return output_path


@app.get("/speech/instant-status")
def speech_instant_status(text: str):
    return {"cached": instant_speech_path(text).exists()}


@app.get("/speech/instant")
async def speech_instant(text: str):
    output_path = await run_in_threadpool(generate_instant_speech, text)
    return FileResponse(
        output_path,
        media_type="audio/mpeg",
        filename="instant_reply.mp3",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/speech/sample")
def speech_sample():
    return FileResponse(
        VOICE_SAMPLE_PATH,
        media_type="audio/wav",
        filename="sudanese_voice_sample.wav",
    )


@app.get("/speech/ack")
def speech_ack():
    return FileResponse(
        Path(__file__).resolve().parent
        / "voice_samples"
        / "generated"
        / "sudanese_ack.wav",
        media_type="audio/wav",
        filename="sudanese_ack.wav",
    )


@app.get("/speech/stream")
def speech_stream(text: str, style: str = "sudanese"):
    return StreamingResponse(
        stream_speech(text, style),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/speech/stream-fast")
def speech_stream_fast(text: str, style: str = "sudanese"):
    return StreamingResponse(
        stream_speech_mp3(text, style),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )

if __name__ == "__main__":
    import uvicorn
    # استخدام المنفذ 5050 لتجنب التداخل مع التطبيقات الشائعة
    print("جاري تشغيل الخادم على http://127.0.0.1:5050")
    uvicorn.run(app, host="127.0.0.1", port=5050, log_level="info")
