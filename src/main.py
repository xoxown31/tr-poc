import os
import sys
import json
from datetime import datetime, timezone

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box

sys.path.insert(0, os.path.dirname(__file__))

from engine.ollama_engine import OllamaEngine
from engine.transition_db import TransitionDB
from engine.executor import LocalExecutor
from reasoner import Reasoner
from problems import PROBLEMS

console = Console()
RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs")

# ── Config ────────────────────────────────────────────────────────────
MODEL       = "gemma4:e4b"
EMBED_MODEL = "nomic-embed-text"
ORTH_THRESHOLD = 0.85
WARM_THRESHOLD = 20   # Phase 2 kicks in after this many DB entries

PASS_BADGE = "[bold white on green] PASS [/bold white on green]"
FAIL_BADGE = "[bold white on red]  FAIL [/bold white on red]"


def badge(ok: bool) -> str:
    return PASS_BADGE if ok else FAIL_BADGE


def print_path(path: list, color: str):
    t = Table(box=box.SIMPLE, show_header=False, show_edge=False, padding=(0, 1))
    t.add_column("d", style="dim", width=4)
    t.add_column("thought")
    for depth, thought in path:
        t.add_row(f"[{color}]d{depth}[/{color}]", thought[:120])
    console.print(t)


def print_code(code: str):
    console.print(Syntax(code.strip(), "python", theme="monokai",
                         line_numbers=True, background_color="default"))


def print_db_stats(db: TransitionDB):
    s = db.stats()
    phase = "[green]WARM[/green]" if s["warm"] else "[yellow]COLD[/yellow]"
    console.print(
        f"  DB  total=[bold]{s['total']}[/bold]  "
        f"pass=[green]{s['pass']}[/green]  "
        f"fail=[red]{s['fail']}[/red]  "
        f"phase={phase}"
    )


def run():
    console.print()
    console.print(Panel(
        "[bold]Transition Replay PoC[/bold]\n"
        "[dim]Off-policy cross-problem reasoning memory  ·  Standard vs Replay[/dim]",
        border_style="cyan", padding=(0, 2)
    ))

    engine   = OllamaEngine(model=MODEL, embed_model=EMBED_MODEL)
    db       = TransitionDB(orthogonality_threshold=ORTH_THRESHOLD, warm_threshold=WARM_THRESHOLD)
    executor = LocalExecutor()
    reasoner = Reasoner(engine=engine, db=db)

    results: list[dict] = []

    for p in PROBLEMS:
        pid   = p["id"]
        prob  = p["problem"]
        tests = p["test"]

        console.print()
        console.print(Rule(f"[bold]Problem {pid} / {len(PROBLEMS)}[/bold]", style="bright_black"))
        console.print(f"[dim]{prob}[/dim]")

        # ── Standard ─────────────────────────────────────────────────
        console.print()
        console.print(Rule("[yellow]Standard  (no DB)[/yellow]", style="bright_black"))

        std_code, std_path = reasoner.run_standard(prob)
        std_res  = executor.execute(std_code, tests)
        std_ok   = std_res["success"]

        print_path(std_path, "yellow")
        print_code(std_code)
        console.print(f"  {badge(std_ok)}")
        if not std_ok and std_res["stderr"]:
            console.print(Panel(std_res["stderr"].strip()[:300],
                                title="[red]Error[/red]", border_style="red dim", padding=(0,1)))

        # Store standard trajectory (Phase 1/2 handled inside)
        added = reasoner.store_trajectory(std_path, std_ok, pid)
        console.print(f"  [dim]→ {added} transition(s) added to DB[/dim]")
        print_db_stats(db)

        # ── Replay ───────────────────────────────────────────────────
        console.print()
        console.print(Rule("[cyan]Replay  (DB injection)[/cyan]", style="bright_black"))

        rep_code, rep_path = reasoner.run_replay(prob)
        rep_res  = executor.execute(rep_code, tests)
        rep_ok   = rep_res["success"]

        print_path(rep_path, "cyan")
        print_code(rep_code)
        console.print(f"  {badge(rep_ok)}")
        if not rep_ok and rep_res["stderr"]:
            console.print(Panel(rep_res["stderr"].strip()[:300],
                                title="[red]Error[/red]", border_style="red dim", padding=(0,1)))

        added = reasoner.store_trajectory(rep_path, rep_ok, pid)
        console.print(f"  [dim]→ {added} transition(s) added to DB[/dim]")
        print_db_stats(db)

        # ── Verdict ───────────────────────────────────────────────────
        if not std_ok and rep_ok:
            console.print("\n  [bold green]✓  Replay fixed it[/bold green]")
        elif std_ok and not rep_ok:
            console.print("\n  [bold red]✗  Replay broke it[/bold red]")

        results.append({"id": pid, "standard": std_ok, "replay": rep_ok})

    # ── Summary ──────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Summary[/bold cyan]", style="cyan"))

    tbl = Table(box=box.ROUNDED, header_style="bold", border_style="bright_black")
    tbl.add_column("Problem", width=10)
    tbl.add_column("Standard", justify="center", width=12)
    tbl.add_column("Replay",   justify="center", width=12)
    tbl.add_column("Δ",        justify="center", width=16)

    std_total = rep_total = 0
    for r in results:
        std_total += r["standard"]
        rep_total += r["replay"]
        if not r["standard"] and r["replay"]:
            delta = "[green]fixed[/green]"
        elif r["standard"] and not r["replay"]:
            delta = "[red]broke[/red]"
        else:
            delta = "[dim]—[/dim]"
        tbl.add_row(f"P{r['id']}", badge(r["standard"]), badge(r["replay"]), delta)

    n = len(results)
    tbl.add_row("", "", "", "")
    tbl.add_row("[bold]Pass Rate[/bold]",
                f"[bold]{std_total}/{n}[/bold]",
                f"[bold]{rep_total}/{n}[/bold]", "")
    console.print(tbl)

    db_s = db.stats()
    console.print(f"\n  Final DB: [bold]{db_s['total']}[/bold] entries  "
                  f"([green]{db_s['pass']}[/green] PASS / [red]{db_s['fail']}[/red] FAIL)")

    # Save results
    os.makedirs(RUNS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = {"timestamp": ts, "model": MODEL, "results": results, "db": db_s}
    path = os.path.join(RUNS_DIR, f"run_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"\n  [dim]Saved → {path}[/dim]")


if __name__ == "__main__":
    run()
