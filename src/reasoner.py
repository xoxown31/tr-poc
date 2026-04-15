import re
from engine.ollama_engine import OllamaEngine
from engine.transition_db import TransitionDB, TransitionEntry


# Fixed depth roles — content varies per problem, role stays consistent
DEPTH_ROLES = {
    1: ("core algorithm",    "What is the core algorithm or approach?"),
    2: ("edge cases",        "What edge cases must be handled?"),
    3: ("implementation",    "How should the implementation be structured step by step?"),
}
MAX_DEPTH = len(DEPTH_ROLES)  # 3 reasoning depths + 1 code generation = 4 LLM calls total


_SYSTEM_REASON = (
    "You are an expert Python programmer. "
    "Answer in ONE concise sentence. "
    "Do NOT write any code or pseudocode — natural language only."
)

_SYSTEM_CODE = (
    "You are an expert Python programmer. "
    "Output ONLY the complete Python function including the def line — no explanation, no markdown fences."
)


def _extract_code(text: str) -> str:
    text = re.sub(r"```python\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _build_context(retrieved: list) -> str:
    if not retrieved:
        return ""
    lines = ["Past experience from similar reasoning directions:"]
    for _, entry in retrieved:
        label = "PASS" if entry.label else "FAIL"
        lines.append(
            f"  [{label}] \"{entry.parent_text[:60]}\" → \"{entry.child_text[:60]}\""
        )
        if entry.reward_text:
            lines.append(f"         Feedback: {entry.reward_text[:200]}")
    return "\n".join(lines)


class Reasoner:
    def __init__(self, engine: OllamaEngine, db: TransitionDB):
        self.engine = engine
        self.db = db

    # ── Standard: no DB ──────────────────────────────────────────────
    def run_standard(self, problem: str) -> tuple[str, list[tuple[int, str]]]:
        path: list[tuple[int, str]] = []
        for depth in range(1, MAX_DEPTH + 1):
            thought = self._gen_thought(problem, path, depth, context="")
            path.append((depth, thought))
        code = self._gen_code(problem, path, context="")
        return _extract_code(code), path

    # ── Replay: DB injection at every step ───────────────────────────
    def run_replay(self, problem: str) -> tuple[str, list[tuple[int, str]]]:
        path: list[tuple[int, str]] = []
        for depth in range(1, MAX_DEPTH + 1):
            parent = path[-1][1] if path else ""
            emb = self.engine.embed(f"{parent}\n{DEPTH_ROLES[depth][0]}")
            retrieved = self.db.retrieve_by_depth(emb, depth=depth, k=3)
            context = _build_context(retrieved)
            thought = self._gen_thought(problem, path, depth, context=context)
            path.append((depth, thought))

        # Code generation with DB context
        parent = path[-1][1] if path else ""
        emb = self.engine.embed(f"{parent}\n[code generation]")
        retrieved = self.db.retrieve_by_depth(emb, depth=MAX_DEPTH, k=3)
        context = _build_context(retrieved)
        code = self._gen_code(problem, path, context=context)
        return _extract_code(code), path

    # ── DB update ────────────────────────────────────────────────────
    def store_trajectory(
        self,
        path: list[tuple[int, str]],
        label: bool,
        problem_id: int,
        verifier_output: str = "",
        problem_text: str = "",
        test_code: str = "",
    ) -> int:
        added = 0
        for i, (depth, thought) in enumerate(path):
            parent = path[i - 1][1] if i > 0 else ""
            trans_key = f"{parent}\n{thought}" if parent else thought
            emb = self.engine.embed(trans_key)

            if self.db.is_warm and not self.db.is_novel(emb):
                continue

            reward_text = self._evaluate(
                thought=thought,
                parent=parent,
                label=label,
                verifier_output=verifier_output,
                problem_text=problem_text,
                test_code=test_code,
            )
            entry = TransitionEntry(
                parent_text=parent,
                child_text=thought,
                label=label,
                reward_text=reward_text,
                depth=depth,
                embedding=emb,
                problem_id=problem_id,
            )
            if self.db.try_add(entry):
                added += 1
        return added

    # ── Internals ────────────────────────────────────────────────────
    def _gen_thought(self, problem: str, path: list, depth: int, context: str) -> str:
        _, question = DEPTH_ROLES[depth]
        path_str = " → ".join(t for _, t in path) if path else "(start)"
        ctx_block = f"\n{context}\n" if context else ""
        prompt = (
            f"Problem: {problem}\n"
            f"Reasoning so far: {path_str}\n"
            f"{ctx_block}"
            f"{question} (one sentence, no code):"
        )
        return self.engine.generate(prompt, system=_SYSTEM_REASON)

    def _gen_code(self, problem: str, path: list, context: str) -> str:
        path_str = " → ".join(t for _, t in path)
        ctx_block = f"\n{context}\n" if context else ""
        prompt = (
            f"Problem: {problem}\n"
            f"Reasoning:\n{path_str}\n"
            f"{ctx_block}"
            f"Write the Python function:"
        )
        return self.engine.generate(prompt, system=_SYSTEM_CODE)

    def _evaluate(
        self,
        thought: str,
        parent: str,
        label: bool,
        verifier_output: str,
        problem_text: str,
        test_code: str,
    ) -> str:
        verdict = "PASSED" if label else "FAILED"
        verifier_block = verifier_output.strip()[:300] if verifier_output.strip() else "No output."
        prompt = (
            f"A reasoning step was taken while solving a coding problem.\n\n"
            f"Problem: {problem_text}\n"
            f"Test code:\n{test_code}\n\n"
            f"Previous reasoning: {parent or '(start)'}\n"
            f"This reasoning step: {thought}\n\n"
            f"Verifier result: {verdict}\n"
            f"Verifier output:\n{verifier_block}\n\n"
            f"In one sentence, explain what this reasoning step did {'right' if label else 'wrong'} "
            f"in the context of the verifier output:"
        )
        commentary = self.engine.generate(
            prompt, system="You are a concise code reasoning evaluator."
        )
        return f"[Verifier: {verdict}] {verifier_block}\n[Evaluator] {commentary.strip()}"
