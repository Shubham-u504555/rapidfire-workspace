import os, requests
PORT = int(os.getenv("PORT", "5005"))
URL = f"http://127.0.0.1:{PORT}/generate"
def ask(prompt: str, max_new_tokens: int = 128):
    r = requests.post(URL, json={"prompt": prompt, "max_new_tokens": max_new_tokens}, timeout=120)
    r.raise_for_status(); return r.json()["text"]
if __name__ == "__main__":
    print(ask("Say hello in exactly five words."))
