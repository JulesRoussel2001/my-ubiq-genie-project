import sys
import os
import argparse
from openai import OpenAI

def request_animation(text, preprompt, prompt_suffix):
    global client
    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": text + prompt_suffix}
    ]
    print(messages, file=sys.stderr)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=10,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def main():
    global client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprompt", type=str, default="")
    parser.add_argument("--prompt_suffix", type=str, default="")
    args = parser.parse_args()

    while True:
        try:
            line = sys.stdin.readline()
            if not line.strip():
                continue
            result = request_animation(line.strip(), args.preprompt, args.prompt_suffix)
            print(result)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
