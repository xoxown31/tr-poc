import numpy as np
import ollama


class OllamaEngine:
    def __init__(self, model: str = "gemma3:4b", embed_model: str = "nomic-embed-text"):
        self.model = model
        self.embed_model = embed_model

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = ollama.chat(model=self.model, messages=messages)
        return resp["message"]["content"].strip()

    def embed(self, text: str) -> np.ndarray:
        resp = ollama.embeddings(model=self.embed_model, prompt=text)
        return np.array(resp["embedding"], dtype=np.float32)
