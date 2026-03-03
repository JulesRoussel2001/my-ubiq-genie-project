import sys, os, argparse
from openai import OpenAI

def request_animation(text, preprompt, prompt_suffix, model, temperature, max_tokens):
    msgs = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": text + prompt_suffix}
    ]
    r = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temperature,   # default 0.0 for reproducibility
        max_tokens=max_tokens,     # default 64 is enough for a label
        top_p=1.0
    )
    return (r.choices[0].message.content or "").strip()

def main():
    global client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--preprompt", type=str, default="")
    ap.add_argument("--prompt_suffix", type=str, default="")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")  # <-- change default here
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=64)
    args = ap.parse_args()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            result = request_animation(
                line, args.preprompt, args.prompt_suffix,
                args.model, args.temperature, args.max_tokens
            )
            print(result)
            sys.stdout.flush()
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
