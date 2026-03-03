#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
from openai import OpenAI
import os
import argparse
import re

def clamp_sentences(text: str, max_sent=3) -> str:
    sents = re.split(r'(?<=[\.\?\!])\s+', (text or "").strip())
    return " ".join([s for s in sents if s][:max_sent]).strip()

def postprocess_reply(text: str) -> str:
    # strip simple roleplay/action markup and clamp length
    text = re.sub(r"\*.*?\*", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    return clamp_sentences(text, 3)

def request_response(client, model, temperature, max_tokens, message_log):
    r = client.chat.completions.create(
        model=model,
        messages=message_log,
        max_tokens=max_tokens,
        temperature=temperature
    )
    out = postprocess_reply(r.choices[0].message.content)
    message_log.append({"role": r.choices[0].message.role, "content": out})
    return message_log

def listen_for_messages(args):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    message_log = [{"role": "system", "content": args.preprompt}]

    while True:
        try:
            line = sys.stdin.buffer.readline()
            if not line or line.isspace():
                continue
            user = line.decode("utf-8").strip() + args.prompt_suffix
            message_log.append({"role": "user", "content": user})
            message_log = request_response(client, args.model, args.temperature, args.max_tokens, message_log)
            print(">" + message_log[-1]["content"])
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprompt", type=str, default="")
    ap.add_argument("--prompt_suffix", type=str, default="")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")  # was gpt-3.5-turbo
    ap.add_argument("--temperature", type=float, default=0.0)    # was 0.7
    ap.add_argument("--max_tokens", type=int, default=160)       # was 1000
    args = ap.parse_args()
    listen_for_messages(args)
