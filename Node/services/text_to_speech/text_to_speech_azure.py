import os, sys, json, re, requests
import azure.cognitiveservices.speech as speechsdk
from xml.sax.saxutils import escape

# --- Azure config ---
SPEECH_KEY = os.environ["SPEECH_KEY"]
SPEECH_REGION = os.environ["SPEECH_REGION"]
VOICE = os.environ.get("SPEECH_VOICE", "en-US-GuyNeural")

speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
)
speech_config.speech_synthesis_voice_name = VOICE
synth = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

# --- Fetch supported styles once ---
def fetch_supported_styles(voice_name:str) -> set[str]:
    url = f"https://{SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/voices/list"
    resp = requests.get(url, headers={"Ocp-Apim-Subscription-Key": SPEECH_KEY}, timeout=10)
    resp.raise_for_status()
    voices = resp.json()
    # voice objects typically have: ShortName, Name, Locale, StyleList, ...
    styles = set()
    for v in voices:
        # Match by ShortName or Name
        if v.get("ShortName") == voice_name or v.get("Name") == voice_name:
            styles = set(v.get("StyleList") or [])  # may be []
            break
    return styles

SUPPORTED_STYLES = fetch_supported_styles(VOICE)

# Conservative fallbacks if requested style is not supported
FALLBACKS = {
    "terrified": "calm",
    "excited": "calm",
    "angry": "neutral",
    "sad": "neutral",
    "cheerful": "neutral",
    "friendly": "neutral",
    "unfriendly": "neutral",
    "whispering": "neutral",
    "shouting": "neutral",
    "customerservice": "neutral",
    "chat": "neutral",
    "assistant": "neutral",
    "newscast-casual": "neutral",
    "newscast-formal": "neutral",
    "narration-professional": "neutral",
}

def best_supported(style:str|None) -> str|None:
    """Return a style that the current voice supports, or None for neutral."""
    s = (style or "neutral").strip().lower()
    if s in ("neutral","none"):
        return None
    if s in SUPPORTED_STYLES:
        return s
    fb = FALLBACKS.get(s, "neutral")
    return None if fb in ("neutral","none") else (fb if fb in SUPPORTED_STYLES else None)

def make_ssml(text: str, voice: str, style: str | None, degree: float = 1.0) -> str:
    text = escape(text or "")
    try:
        degree = float(degree)
    except Exception:
        degree = 1.0

    # Resolve to a supported style (or None → neutral)
    style_used = best_supported(style)

    if not style_used:
        # Neutral: no express-as
        return f"""<speak version="1.0"
    xmlns="http://www.w3.org/2001/10/synthesis"
    xmlns:mstts="https://www.w3.org/2001/mstts"
    xml:lang="en-US">
  <voice name="{voice}">{text}</voice>
</speak>"""

    return f"""<speak version="1.0"
    xmlns="http://www.w3.org/2001/10/synthesis"
    xmlns:mstts="https://www.w3.org/2001/mstts"
    xml:lang="en-US">
  <voice name="{voice}">
    <mstts:express-as style="{style_used}" styledegree="{degree}">{text}</mstts:express-as>
  </voice>
</speak>"""

def synth_bytes(ssml: str):
    result = synth.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data or b""
    # Graceful error surface (very rare once we pre-check styles)
    if result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.CancellationDetails(result)
        err = getattr(details, "error_details", "") or ""
        raise RuntimeError(f"Canceled: {details.reason}; {err}")
    raise RuntimeError(f"Synthesis failed: {result.reason}")

def read_line():
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    return line.decode("utf-8", errors="ignore").strip()

# stdin loop (unchanged protocol)
while True:
    raw = read_line()
    if raw is None:
        break
    if not raw:
        continue

    text = None
    style = "neutral"
    degree = 1.0

    try:
        obj = json.loads(raw)
        text = (obj.get("text") or "").strip()
        style = (obj.get("style") or "neutral").strip()
        degree = obj.get("degree") or 1.0
    except Exception:
        text = raw

    if not text:
        continue

    try:
        ssml = make_ssml(text, VOICE, style, degree)
        pcm = synth_bytes(ssml)
    except Exception:
        # last-resort neutral
        ssml = make_ssml(text, VOICE, None, 1.0)
        try:
            pcm = synth_bytes(ssml)
        except Exception:
            pcm = b""

    sys.stdout.write(f"LEN:{len(pcm)}\n")
    sys.stdout.buffer.write(pcm)
    sys.stdout.flush()
