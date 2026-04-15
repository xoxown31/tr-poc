# Transition Replay PoC

Off-policy cross-problem reasoning memory for LLM code generation.

## Setup

```bash
conda env create -f environment.yml
conda activate tr-poc
```

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

## Run

```bash
cd src
python main.py
```

## Idea

Each reasoning step `(parent → child)` is stored as a labeled transition `(PASS/FAIL)` in a vector DB.
New problems retrieve similar past transitions and inject them as context — off-policy replay.

**Phase 1 (cold)**: all novel transitions verified and stored.
**Phase 2 (warm)**: orthogonality gate first — only novel patterns trigger verification.
