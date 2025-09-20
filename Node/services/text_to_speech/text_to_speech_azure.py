import os
import sys
import azure.cognitiveservices.speech as speechsdk

# Azure config
speech_config = speechsdk.SpeechConfig(
    subscription=os.environ.get('SPEECH_KEY'),
    region=os.environ.get('SPEECH_REGION')
)

# Choose ONE: match this with app.ts SAMPLE_RATE_HZ
# speech_config.set_speech_synthesis_output_format(
#     speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
# )
speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
)

speech_config.speech_synthesis_voice_name = 'en-US-GuyNeural'
synth = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

# Protocol:
# For each input line (one sentence), write:
#   b"LEN:<n>\n"           (ASCII header)
#   <n raw PCM bytes>      (binary)
# No flush() needed if parent launches Python with -u.

for line in sys.stdin:
    text = line.strip()
    if not text:
        continue
    result = synth.speak_text_async(text).get()
    pcm = result.audio_data or b""

    sys.stdout.write(f"LEN:{len(pcm)}\n")   # ASCII header
    # Do NOT print anything else to stdout.
    sys.stdout.buffer.write(pcm)            # raw bytes immediately after header
    # No explicit flush needed when parent uses `python -u`
