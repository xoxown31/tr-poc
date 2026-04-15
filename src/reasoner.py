import re
from engine.ollama_engine import OllamaEngine
from engine.transition_db import TransitionDB, TransitionEntry


K_CANDIDATES = 3
MAX_DEPTH = 3


_SYSTEM = (
    "You are an expert Python programmer. "
    "Respond concisely. When asked for a reasoning step, output ONE short sentence. "
    "When asked for code, output ONLY the Python function — no explanation, no markdown fences."
)


def _extract_code(text: str) -> str:
    # Strip markdown fences if present
    text = re.sub(r"```python\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _build_context(retrieved: list) -> str:
    if not retrieved:
        return ""
    lines = ["Past experience from similar reasoning directions:"]
    for sim, entry in retrieved:
        label = "PASS" if entry.label else "FAIL"
        lines.append(
            f"  [{label}] \"{entry.parent_text[:60]}\" → \"{entry.child_text[:60]}\""
        )
    return "\n".join(lines)


class Reasoner:
    def __init__(self, engine: OllamaEngine, db: TransitionDB):
        self.engine = engine
        self.db = db

    # ── Standard: no DB ──────────────────────────────────────────────
    def run_standard(self, problem: str) -> tuple[str, list[tuple[int, str]]]:
        path: list[tuple[int, str]] = []
        for depth in range(1, MAX_DEPTH + 1):
            thought = self._gen_thought(problem, path, context="")
            path.append((depth, thought))
        code = self._gen_code(problem, path, context="")
        return _extract_code(code), path

    # ── Replay: DB injection ──────────────────────────────────────────
    def run_replay(self, problem: str) -> tuple[str, list[tuple[int, str]]]:
        path: list[tuple[int, str]] = []
        for depth in range(1, MAX_DEPTH + 1):
            parent = path[-1][1] if path else ""
            # Sample k candidates, pick best via DB signal
            best_thought, best_score = None, -1.0
            for _ in range(K_CANDIDATES):
                cand = self._gen_thought(problem, path, context="")
                score, context = self._score_candidate(parent, cand)
                if score > best_score:
                    best_score, best_thought = score, cand
            path.append((depth, best_thought))

        # Final code generation with full context
        parent = path[-1][1] if path else ""
        trans_key = f"{parent}\n[code generation]"
        emb = self.engine.embed(trans_key)
        retrieved = self.db.retrieve(emb, k=3)
        context = _build_context(retrieved)
        code = self._gen_code(problem, path, context=context)
        return _extract_code(code), path

    # ── DB update ────────────────────────────────────────────────────
    def store_trajectory(self, path: list[tuple[int, str]], label: bool, problem_id: int) -> int:
        """
        Phase 1 (cold): store all novel transitions regardless.
        Phase 2 (warm): orthogonality check doubles as gate — try_add handles it.
        Returns number of new entries added.
        """
        added = 0
        for i, (depth, thought) in enumerate(path):
            parent = path[i - 1][1] if i > 0 else ""
            trans_key = f"{parent}\n{thought}" if parent else thought
            emb = self.engine.embed(trans_key)
            entry = TransitionEntry(
                parent_text=parent,
                child_text=thought,
                label=label,
                embedding=emb,
                problem_id=problem_id,
            )
            if self.db.try_add(entry):
                added += 1
        return added

    # ── Internals ────────────────────────────────────────────────────
    def _gen_thought(self, problem: str, path: list, context: str) -> str:
        path_str = " → ".join(t for _, t in path) if path else "(start)"
        ctx_block = f"\n{context}\n" if context else ""
        prompt = (
            f"Problem: {problem}\n"
            f"Reasoning so far: {path_str}\n"
            f"{ctx_block}"
            f"Next reasoning step (one sentence):"
        )
        return self.engine.generate(prompt, system=_SYSTEM)

    def _gen_code(self, problem: str, path: list, context: str) -> str:
        path_str = " → ".join(t for _, t in path)
        ctx_block = f"\n{context}\n" if context else ""
        prompt = (
            f"Problem: {problem}\n"
            f"Reasoning: {path_str}\n"
            f"{ctx_block}"
            f"Write the Python function:"
        )
        return self.engine.generate(prompt, system=_SYSTEM)

    def _score_candidate(self, parent: str, candidate: str) -> tuple[float, str]:
        trans_key = f"{parent}\n{candidate}" if parent else candidate
        emb = self.engine.embed(trans_key)
        retrieved = self.db.retrieve(emb, k=3)
        context = _build_context(retrieved)

        if not retrieved:
            return 0.5, context

        # Score: mean similarity weighted by label (+1 PASS, -1 FAIL)
        score = 0.5
        for sim, entry in retrieved:
            score += sim * (0.3 if entry.label else -0.3)
        return max(0.0, min(1.0, score)), context
