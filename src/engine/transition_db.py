from dataclasses import dataclass, field
import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class TransitionEntry:
    parent_text: str
    child_text: str
    label: bool          # True = PASS, False = FAIL
    embedding: np.ndarray
    problem_id: int


class TransitionDB:
    """
    Off-policy transition memory.
    Stores (parent→child) reasoning transitions labeled PASS/FAIL.
    Orthogonality gate: only admits novel transitions (max_sim < threshold).

    Phase 1 (cold start): is_warm=False → verify all, store all novel
    Phase 2 (warm):       is_warm=True  → orthogonality check first,
                                          only novel ones proceed to verification
    """

    def __init__(self, orthogonality_threshold: float = 0.85, warm_threshold: int = 20):
        self.entries: list[TransitionEntry] = []
        self.orth_threshold = orthogonality_threshold
        self.warm_threshold = warm_threshold  # DB size at which Phase 2 kicks in

    @property
    def is_warm(self) -> bool:
        return len(self.entries) >= self.warm_threshold

    def is_novel(self, embedding: np.ndarray) -> bool:
        if not self.entries:
            return True
        sims = [cosine_sim(embedding, e.embedding) for e in self.entries]
        return max(sims) < self.orth_threshold

    def try_add(self, entry: TransitionEntry) -> bool:
        """Returns True if added (novel), False if skipped (redundant)."""
        if self.is_novel(entry.embedding):
            self.entries.append(entry)
            return True
        return False

    def retrieve(self, query: np.ndarray, k: int = 3) -> list[tuple[float, TransitionEntry]]:
        if not self.entries:
            return []
        scored = [(cosine_sim(query, e.embedding), e) for e in self.entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def stats(self) -> dict:
        n_pass = sum(1 for e in self.entries if e.label)
        n_fail = len(self.entries) - n_pass
        return {"total": len(self.entries), "pass": n_pass, "fail": n_fail, "warm": self.is_warm}
