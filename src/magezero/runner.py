"""
runner.py — orchestrates the full MageZero training pipeline.

For each generation:
  1. Resolve curriculum settings
  2. For each opponent (sequential):
       reserve session ids for primary + opponent
       build a mutated game.yml
       start servers (skipped for offline)
       launch JVM, write hdf5 files, stop servers
  3. Optional: dataset_stats on new data
  4. Optional: eval previous model on new data
  5. Move testing/ → training/
  6. Archive out-of-window training files
  7. Train (50 epochs bootstrap, 10 epochs from checkpoint)
  8. Restore archived files
  9. Record gen completion in the run file
"""
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from magezero.util.config import (
    CurriculumConfig,
    GenSettings,
    Opponent,
    RunConfig,
    load_all,
    resolve_gen,
)


# ─── constants ───────────────────────────────────────────────

PRIMARY_PORT = 50052
OPPONENT_PORT = 50053
TMP_GAME_YML = ".mz_tmp/game.yml"
RUNS_DIR = Path("runs")
SRC = "src/magezero"
PYTHON = sys.executable
EPOCHS_BOOTSTRAP = 50
EPOCHS_ONLINE = 10


# ─── deck-level state (session counter) ──────────────────────

def deck_state_path(deck: str) -> Path:
    return Path("models") / deck / "state.json"


def next_session_id(deck: str) -> int:
    """Reserve and return the next session id for this deck."""
    p = deck_state_path(deck)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(p.read_text()) if p.exists() else {"next_session_id": 1}
    sid = state["next_session_id"]
    state["next_session_id"] = sid + 1
    p.write_text(json.dumps(state, indent=2))
    return sid


# ─── run folders (history + active state) ───────────────────

def find_active_run(deck: str, version: int) -> Optional[Path]:
    if not RUNS_DIR.exists():
        return None
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        json_file = d / "run.json"
        if not json_file.exists():
            continue
        data = json.loads(json_file.read_text())
        if (data.get("completed_at") is None
                and data.get("abandoned_at") is None
                and data["primary"]["deck"] == deck
                and data["primary"]["version"] == version):
            return d
    return None


def create_run_dir(run: RunConfig) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir()
    data = {
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "abandoned_at": None,
        "primary": {"deck": run.deck, "version": run.version},
        "opponents": [
            {"deck": o.deck, "mode": o.mode, "version": o.version}
            for o in run.opponents
        ],
        "run_config_snapshot": {
            "generations": run.generations,
            "games_per_gen": run.games_per_gen,
            "replay_buffer_gens": run.replay_buffer_gens,
            "start_from_version": run.start_from_version,
        },
        "current_gen": 0,
        "stage": "init",
        "gens": {},
    }
    (run_dir / "run.json").write_text(json.dumps(data, indent=2))
    return run_dir


def update_run(run_dir: Path, **fields) -> dict:
    json_file = run_dir / "run.json"
    data = json.loads(json_file.read_text())
    data.update(fields)
    json_file.write_text(json.dumps(data, indent=2))
    return data


def record_gen(run_dir: Path, gen: int, settings: GenSettings,
               primary_sessions: dict, opponent_sessions: dict) -> None:
    json_file = run_dir / "run.json"
    data = json.loads(json_file.read_text())
    data["gens"][str(gen)] = {
        "primary_sessions": primary_sessions,
        "opponent_sessions": opponent_sessions,
        "settings": {
            "td_discount": settings.td_discount,
            "prior_temperature": settings.prior_temperature,
            "priors": {
                "binary": settings.priors.binary,
                "priority": settings.priors.priority,
                "target": settings.priors.target,
                "opponent": settings.priors.opponent,
            },
        },
        "completed_at": datetime.now().isoformat(),
    }
    json_file.write_text(json.dumps(data, indent=2))


# ─── version helpers ─────────────────────────────────────────

def latest_version(deck: str) -> Optional[int]:
    d = Path("models") / deck
    if not d.exists():
        return None
    versions = []
    for sub in d.iterdir():
        if sub.is_dir() and sub.name.startswith("ver"):
            try:
                v = int(sub.name[3:])
                if (sub / "model.pt.gz").exists():
                    versions.append(v)
            except ValueError:
                pass
    return max(versions) if versions else None


def has_checkpoint(deck: str, version: int) -> bool:
    return (Path("models") / deck / f"ver{version}" / "model.pt.gz").exists()


def copy_starting_checkpoint(run: RunConfig) -> None:
    """If start_from_version is set, seed the new ver folder from the source."""
    if run.start_from_version is None:
        return
    src = Path("models") / run.deck / f"ver{run.start_from_version}"
    dst = Path("models") / run.deck / f"ver{run.version}"
    if dst.exists() and (dst / "model.pt.gz").exists():
        return  # already seeded (resume case)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt.gz", "ignore.roar"):
        f = src / name
        if f.exists():
            shutil.copy(f, dst / name)


# ─── data file path helpers ──────────────────────────────────

def primary_file(deck: str, version: int, sid: int, opponent: str) -> Path:
    name = f"session{sid}_{deck}_vs_{opponent}.hdf5"
    return Path("data") / deck / f"ver{version}" / "testing" / name


def opponent_file(opp: str, opp_ver: int, sid: int, primary: str) -> Path:
    name = f"session{sid}_{opp}_vs_{primary}.hdf5"
    return Path("data") / opp / f"ver{opp_ver}" / "archive" / name


def parse_session_id(filename: str) -> Optional[int]:
    """session{N}_..."""
    try:
        return int(filename.split("_")[0].replace("session", ""))
    except (ValueError, IndexError):
        return None


# ─── game.yml mutation ───────────────────────────────────────

def build_game_yml(base_path: str, settings: GenSettings, run: RunConfig,
                   opp: Opponent, primary_out: Path, opponent_out: Path,
                   primary_offline: bool, opp_offline: bool) -> str:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    decks_dir = Path("xmage/decks").resolve()

    pa = cfg["player_a"]
    pa["deckPath"] = str(decks_dir / f"{run.deck}.dck")
    pa["output_file"] = str(primary_out.resolve())
    pa["mcts"]["offline_mode"] = primary_offline
    pa["mcts"]["td_discount"] = settings.td_discount
    pa["priors"]["prior_temperature"] = settings.prior_temperature
    pa["priors"]["binary"] = settings.priors.binary
    pa["priors"]["priority"] = settings.priors.priority
    pa["priors"]["target"] = settings.priors.target
    pa["priors"]["opponent"] = settings.priors.opponent

    pb = cfg["player_b"]
    pb["deckPath"] = str(decks_dir / f"{opp.deck}.dck")
    pb["output_file"] = str(opponent_out.resolve())
    pb["type"] = "minimax" if opp.mode == "minimax" else "mcts"
    pb["mcts"]["offline_mode"] = opp_offline

    cfg["training"]["games"] = run.games_per_gen
    cfg["server"]["port"] = PRIMARY_PORT
    cfg["server"]["opponent_port"] = OPPONENT_PORT

    out = Path(TMP_GAME_YML)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(out)


# ─── subprocess wrappers ─────────────────────────────────────

def start_server(deck: str, version: int, port: int, run_dir: Path) -> subprocess.Popen:
    print(f"[server] start {deck} v{version} on :{port}")
    log_path = run_dir / f"server_{port}.log"
    log_file = open(log_path, "a")
    log_file.write(f"\n=== START {datetime.now().isoformat()} deck={deck} v{version} ===\n")
    log_file.flush()

    proc = subprocess.Popen(
        [PYTHON, f"{SRC}/server.py",
         "--deck", deck, "--version", str(version), "--port", str(port)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    proc._mz_log_file = log_file

    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            raise RuntimeError("server process exited before becoming ready")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            print(f"[server] ready on :{port}")
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    log_file.close()
    raise TimeoutError(f"server on :{port} did not come up within 600s")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    if hasattr(proc, "_mz_log_file"):
        proc._mz_log_file.close()


def launch_jvm(game_yml_path: str) -> None:
    print(f"[jvm] launching with {game_yml_path}")
    yml = str(Path(game_yml_path).resolve())
    if sys.platform == "win32":
        cmd = ["cmd", "/c", "xmage\\mz-xmage.bat", yml]
    else:
        cmd = ["sh", "xmage/mz-xmage.sh", yml]
    subprocess.run(cmd, check=True)


def launch_jvm_capture(game_yml_path: str) -> str:
    """Launch JVM and capture its stdout+stderr for metric parsing."""
    print(f"[jvm] launching with {game_yml_path}")
    yml = str(Path(game_yml_path).resolve())
    if sys.platform == "win32":
        cmd = ["cmd", "/c", "xmage\\mz-xmage.bat", yml]
    else:
        cmd = ["sh", "xmage/mz-xmage.sh", yml]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print stderr to console so errors are visible
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd,
                                            output=result.stdout, stderr=result.stderr)
    return result.stdout


# ─── JVM output parsing ─────────────────────────────────────

# Patterns from ComputerPlayerMCTS2.applyMCTS and ParallelDataGenerator
_RE_TOTAL_SIMS = re.compile(
    r"Total: simulated (\d+) evaluations in ([\d.]+) seconds - Average: ([\d.]+)"
)
_RE_GAME_DONE = re.compile(r"Game #(\d+) completed successfully")
_RE_SUCCESSFUL = re.compile(r"Successful: (\d+)")
_RE_FAILED = re.compile(r"Failed: (\d+)")
_RE_WIN_RATE = re.compile(r"Player A win rate: ([\d.]+)% \((\d+)/(\d+)\)")
_RE_DATA_GEN_START = re.compile(r"STARTING DATA GENERATION")
# Turn numbers appear as [N:Phase:Step] in log lines
_RE_TURN_NUM = re.compile(r"\[(\d+):[A-Za-z]")


def parse_jvm_metrics(output: str) -> dict:
    """Parse JVM stdout to extract generation performance metrics."""
    # Collect all 'Total:' lines — each is the cumulative counter at that point.
    # The last Total: line per thread shows the final sims/sec for that thread.
    all_averages = []
    total_evals = 0
    total_seconds = 0.0
    for m in _RE_TOTAL_SIMS.finditer(output):
        evals = int(m.group(1))
        secs = float(m.group(2))
        avg = float(m.group(3))
        all_averages.append(avg)
        # Keep the highest cumulative eval count (last per-thread line)
        if evals > total_evals:
            total_evals = evals
            total_seconds = secs

    # Game completion stats
    games_done = len(_RE_GAME_DONE.findall(output))
    successful_m = _RE_SUCCESSFUL.search(output)
    failed_m = _RE_FAILED.search(output)
    win_rate_m = _RE_WIN_RATE.search(output)

    games_successful = int(successful_m.group(1)) if successful_m else 0
    games_failed = int(failed_m.group(1)) if failed_m else 0
    wins = int(win_rate_m.group(2)) if win_rate_m else 0
    total_played = int(win_rate_m.group(3)) if win_rate_m else games_successful

    # Extract turn numbers from [N:Phase:Step] patterns
    turn_numbers = [int(m.group(1)) for m in _RE_TURN_NUM.finditer(output)]
    max_turn = max(turn_numbers) if turn_numbers else 0
    # Approximate total turns: sum of unique turn numbers seen (rough proxy)
    # More accurate: count distinct turn numbers per "STARTING DATA" block
    # For now, use max turn * games as an upper-bound proxy
    total_turns_approx = 0
    if games_done > 0 and turn_numbers:
        # Split by STARTING DATA blocks to get per-opponent-session turn counts
        # Simpler: average the max turn across all games
        # Use the set of all turn numbers as a rough indicator
        total_turns_approx = max_turn * games_done  # rough approximation

    result = {
        "games_completed": games_done,
        "games_successful": games_successful,
        "games_failed": games_failed,
        "wins": wins,
        "losses": total_played - wins,
        "win_rate_pct": float(win_rate_m.group(1)) if win_rate_m else None,
        "total_mcts_evals": total_evals,
        "mcts_sims_per_sec_final": round(all_averages[-1], 2) if all_averages else 0,
        "mcts_sims_per_sec_mean": round(sum(all_averages) / len(all_averages), 2) if all_averages else 0,
        "max_turn": max_turn,
        "total_turns_approx": total_turns_approx,
        "avg_turns_per_game": round(total_turns_approx / games_done, 1) if games_done > 0 else 0,
    }
    return result


def run_train(deck: str, version: int, epochs: int, use_checkpoint: bool,
              run_dir: Path, gen: int) -> None:
    cmd = [PYTHON, f"{SRC}/train.py",
           "--deck", deck, "--version", str(version),
           "--epochs", str(epochs)]
    if use_checkpoint:
        cmd.append("--checkpoint")
    log_path = run_dir / "train.log"
    with open(log_path, "a") as f:
        f.write(f"\n=== GEN {gen} TRAIN {datetime.now().isoformat()} ===\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def run_test(deck: str, version: int, run_dir: Path, gen: int) -> None:
    log_path = run_dir / "test.log"
    with open(log_path, "a") as f:
        f.write(f"\n=== GEN {gen} TEST {datetime.now().isoformat()} ===\n")
        f.flush()
        subprocess.run(
            [PYTHON, f"{SRC}/test.py", "--deck", deck, "--version", str(version)],
            check=True, stdout=f, stderr=subprocess.STDOUT,
        )


def run_dataset_stats(deck: str, version: int, split: str,
                      run_dir: Path, gen: int) -> None:
    log_path = run_dir / "dataset_stats.log"
    with open(log_path, "a") as f:
        f.write(f"\n=== GEN {gen} DATASET_STATS {datetime.now().isoformat()} ===\n")
        f.flush()
        subprocess.run(
            [PYTHON, f"{SRC}/dataset_stats.py",
             "--deck", deck, "--version", str(version), "--split", split],
            check=True, stdout=f, stderr=subprocess.STDOUT,
        )


# ─── data movement ───────────────────────────────────────────

def move_testing_to_training(deck: str, version: int) -> None:
    src = Path("data") / deck / f"ver{version}" / "testing"
    dst = Path("data") / deck / f"ver{version}" / "training"
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    for f in src.glob("*.hdf5"):
        shutil.move(str(f), str(dst / f.name))


def archive_out_of_window(deck: str, version: int, gens: dict,
                          current_gen: int, window: int) -> list[Path]:
    """Move training files outside the replay window into archive/. Returns moved paths."""
    cutoff = current_gen - window + 1
    if cutoff <= 0:
        return []

    archive_ids: set[int] = set()
    for g_str, g_data in gens.items():
        if int(g_str) < cutoff:
            for sids in g_data.get("primary_sessions", {}).values():
                archive_ids.update(sids)

    if not archive_ids:
        return []

    training = Path("data") / deck / f"ver{version}" / "training"
    archive = Path("data") / deck / f"ver{version}" / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    moved = []
    for f in training.glob("*.hdf5"):
        sid = parse_session_id(f.name)
        if sid is not None and sid in archive_ids:
            dst = archive / f.name
            shutil.move(str(f), str(dst))
            moved.append(dst)
    return moved


def restore_from_archive(deck: str, version: int, files: list[Path]) -> None:
    training = Path("data") / deck / f"ver{version}" / "training"
    for f in files:
        shutil.move(str(f), str(training / f.name))


# ─── training curve ──────────────────────────────────────────

CURVE_HEADER = ("gen,opponent,games,wins,losses,win_rate_pct,"
                "wall_sec,games_per_hour,sims_per_sec_avg,sims_per_sec_end,"
                "inferences_per_sec,avg_turns_per_game,max_turn,"
                "total_turns_approx,mode\n")


def append_training_curve(run_dir: Path, gen: int, opponent: str,
                          jvm: dict, wall_sec: float,
                          srv: Optional[dict], offline: bool) -> None:
    """Append one row per opponent to the training curve CSV."""
    csv_path = run_dir / "training_curve.csv"
    if not csv_path.exists():
        csv_path.write_text(CURVE_HEADER)

    games = jvm.get("games_successful", 0)
    wins = jvm.get("wins", 0)
    losses = jvm.get("losses", 0)
    wr = jvm.get("win_rate_pct", 0) or 0
    gph = round(games / wall_sec * 3600, 1) if wall_sec > 0 and games > 0 else 0
    sims_avg = jvm.get("mcts_sims_per_sec_mean", 0)
    sims_end = jvm.get("mcts_sims_per_sec_final", 0)
    inf_sec = srv.get("inferences_per_sec", 0) if srv else 0
    avg_turns = jvm.get("avg_turns_per_game", 0)
    max_turn = jvm.get("max_turn", 0)
    total_turns = jvm.get("total_turns_approx", 0)
    mode = "offline" if offline else "online"

    row = (f"{gen},{opponent},{games},{wins},{losses},{wr},"
           f"{wall_sec},{gph},{sims_avg},{sims_end},"
           f"{inf_sec},{avg_turns},{max_turn},"
           f"{total_turns},{mode}\n")
    with open(csv_path, "a") as f:
        f.write(row)


# ─── server metrics ──────────────────────────────────────────

def fetch_server_metrics(port: int) -> Optional[dict]:
    """Fetch metrics from a running inference server."""
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
        import json as _json
        return _json.loads(resp.read())
    except Exception:
        return None


def reset_server_metrics(port: int) -> None:
    """Reset metrics counters on a running inference server."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics/reset", method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# ─── benchmark ───────────────────────────────────────────────

def run_benchmark(run: RunConfig, curriculum: CurriculumConfig,
                  base_game_yml: str = "configs/game.yml") -> dict:
    """Run a single online generation with metric capture.

    Requires an existing checkpoint for the deck/version so the inference
    server can be started. Does not modify training state — output goes
    to a temp directory that is cleaned up afterward.

    Returns a dict with:
      - jvm: parsed JVM metrics (sims/sec, games, win rate)
      - server: inference server metrics (throughput, latency, batch distribution)
      - wall_time_sec: total wall time for the generation
    """
    if not has_checkpoint(run.deck, run.version):
        raise RuntimeError(
            f"No checkpoint found at models/{run.deck}/ver{run.version}/model.pt.gz. "
            f"Run at least one training generation first:\n"
            f"  mz train --run {run.curriculum_path.replace('curriculum', 'run')}\n"
            f"Or use the smoke config:\n"
            f"  mz train --run configs/run.baylen-smoke.yml"
        )

    settings = resolve_gen(curriculum, 0)
    opp = run.opponents[0]
    opp_ver = opp.version if opp.version is not None else latest_version(opp.deck)
    if opp_ver is None:
        opp_ver = 1
    opp_offline = (opp.mode == "mcts") and not has_checkpoint(opp.deck, opp_ver)

    # Use temp paths so we don't pollute real training data
    tmp_dir = Path(".mz_tmp/benchmark")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    primary_path = tmp_dir / "primary.hdf5"
    opponent_path = tmp_dir / "opponent.hdf5"

    game_yml = build_game_yml(
        base_game_yml, settings, run, opp,
        primary_path, opponent_path,
        primary_offline=False, opp_offline=opp_offline,
    )

    # Create a temporary run dir for server logs
    run_dir = Path(".mz_tmp/benchmark_run")
    run_dir.mkdir(parents=True, exist_ok=True)

    servers = []
    servers.append(start_server(run.deck, run.version, PRIMARY_PORT, run_dir))
    reset_server_metrics(PRIMARY_PORT)
    if opp.mode == "mcts" and not opp_offline:
        servers.append(start_server(opp.deck, opp_ver, OPPONENT_PORT, run_dir))
        reset_server_metrics(OPPONENT_PORT)

    wall_start = time.perf_counter()
    try:
        jvm_output = launch_jvm_capture(game_yml)
    finally:
        # Fetch server metrics before stopping
        server_metrics = fetch_server_metrics(PRIMARY_PORT)
        for s in servers:
            stop_server(s)
    wall_end = time.perf_counter()

    jvm_metrics = parse_jvm_metrics(jvm_output)
    wall_time = round(wall_end - wall_start, 2)

    # Compute games/hour
    games_per_hour = 0
    if wall_time > 0 and jvm_metrics["games_successful"] > 0:
        games_per_hour = round(jvm_metrics["games_successful"] / wall_time * 3600, 1)

    # Clean up temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    report = {
        "wall_time_sec": wall_time,
        "games_per_hour": games_per_hour,
        "jvm": jvm_metrics,
        "server": server_metrics,
        "config": {
            "deck": run.deck,
            "version": run.version,
            "opponent": opp.deck,
            "opponent_mode": opp.mode,
            "games": run.games_per_gen,
            "offline": False,
        },
    }
    return report


# ─── main pipeline ───────────────────────────────────────────

def run_pipeline(run: RunConfig, curriculum: CurriculumConfig,
                 base_game_yml: str = "configs/game.yml") -> None:
    # ── resume detection ──
    active = find_active_run(run.deck, run.version)
    if active:
        ans = input(f"Active run found: {active.name}. Resume? [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            run_dir = active
            start_gen = json.loads((active / "run.json").read_text())["current_gen"]
            print(f"Resuming from gen {start_gen}")
        else:
            update_run(active, abandoned_at=datetime.now().isoformat())
            copy_starting_checkpoint(run)
            run_dir = create_run_dir(run)
            start_gen = 0
    else:
        copy_starting_checkpoint(run)
        run_dir = create_run_dir(run)
        start_gen = 0

    # ── gen loop ──
    for gen in range(start_gen, run.generations):
        print(f"\n========== GEN {gen} ==========")
        update_run(run_dir, current_gen=gen, stage="generate")
        settings = resolve_gen(curriculum, gen)
        bootstrap = not has_checkpoint(run.deck, run.version)

        primary_sessions: dict[str, list[int]] = {}
        opponent_sessions: dict[str, list[int]] = {}

        for opp in run.opponents:
            print(f"\n[gen {gen}] vs {opp.deck} ({opp.mode})")

            opp_ver = opp.version if opp.version is not None else latest_version(opp.deck)
            if opp_ver is None:
                opp_ver = 1
            opp_offline = (opp.mode == "mcts") and not has_checkpoint(opp.deck, opp_ver)
            primary_offline = bootstrap

            primary_sid = next_session_id(run.deck)
            opponent_sid = next_session_id(opp.deck)
            primary_path = primary_file(run.deck, run.version, primary_sid, opp.deck)
            opponent_path = opponent_file(opp.deck, opp_ver, opponent_sid, run.deck)
            primary_path.parent.mkdir(parents=True, exist_ok=True)
            opponent_path.parent.mkdir(parents=True, exist_ok=True)

            game_yml = build_game_yml(
                base_game_yml, settings, run, opp,
                primary_path, opponent_path, primary_offline, opp_offline,
            )

            servers = []
            if not primary_offline:
                servers.append(start_server(run.deck, run.version, PRIMARY_PORT, run_dir))
                reset_server_metrics(PRIMARY_PORT)
            if opp.mode == "mcts" and not opp_offline:
                servers.append(start_server(opp.deck, opp_ver, OPPONENT_PORT, run_dir))

            gen_wall_start = time.perf_counter()
            try:
                jvm_output = launch_jvm_capture(game_yml)
            finally:
                gen_wall_end = time.perf_counter()
                # Fetch server metrics before stopping
                srv_metrics = None
                if not primary_offline:
                    srv_metrics = fetch_server_metrics(PRIMARY_PORT)
                for s in servers:
                    stop_server(s)

            gen_wall_sec = round(gen_wall_end - gen_wall_start, 2)
            jvm_metrics = parse_jvm_metrics(jvm_output)

            # Print per-opponent summary
            wins = jvm_metrics.get("wins", 0)
            losses = jvm_metrics.get("losses", 0)
            wr = jvm_metrics.get("win_rate_pct", 0) or 0
            games_ok = jvm_metrics.get("games_successful", 0)
            avg_turns = jvm_metrics.get("avg_turns_per_game", 0)
            gph = round(games_ok / gen_wall_sec * 3600, 1) if gen_wall_sec > 0 and games_ok > 0 else 0
            sims = jvm_metrics.get("mcts_sims_per_sec_mean", 0)
            print(f"[gen {gen}] vs {opp.deck}: {wins}W/{losses}L ({wr:.1f}%) | "
                  f"{games_ok} games in {gen_wall_sec}s ({gph} games/hr) | "
                  f"{avg_turns} avg turns/game | {sims} sims/sec")
            if srv_metrics:
                print(f"[gen {gen}] server: {srv_metrics.get('inferences_per_sec', '?')} inferences/sec, "
                      f"p50={srv_metrics.get('latency_p50_ms', '?')}ms, "
                      f"p95={srv_metrics.get('latency_p95_ms', '?')}ms")

            # Append to training curve CSV
            append_training_curve(run_dir, gen, opp.deck, jvm_metrics,
                                  gen_wall_sec, srv_metrics, primary_offline)

            primary_sessions.setdefault(opp.deck, []).append(primary_sid)
            opponent_sessions.setdefault(opp.deck, []).append(opponent_sid)

        # ── analyze new data ──
        if run.training.analyze_dataset:
            update_run(run_dir, stage="analyze")
            run_dataset_stats(run.deck, run.version, "testing", run_dir, gen)

        # ── eval previous model on new data ──
        if run.training.eval_previous_model and gen > 0:
            update_run(run_dir, stage="eval")
            run_test(run.deck, run.version, run_dir, gen)

        # ── move testing → training ──
        update_run(run_dir, stage="move")
        move_testing_to_training(run.deck, run.version)

        # ── archive out-of-window data ──
        update_run(run_dir, stage="train")
        data = json.loads((run_dir / "run.json").read_text())
        provisional = dict(data["gens"])
        provisional[str(gen)] = {"primary_sessions": primary_sessions}
        archived = archive_out_of_window(
            run.deck, run.version, provisional, gen, run.replay_buffer_gens,
        )

        # ── train ──
        try:
            epochs = EPOCHS_BOOTSTRAP if bootstrap else EPOCHS_ONLINE
            run_train(run.deck, run.version, epochs, use_checkpoint=not bootstrap,
                      run_dir=run_dir, gen=gen)
        finally:
            restore_from_archive(run.deck, run.version, archived)

        record_gen(run_dir, gen, settings, primary_sessions, opponent_sessions)

    update_run(run_dir, completed_at=datetime.now().isoformat(), stage="done")
    print(f"\n✓ Run complete: {run_dir.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="configs/run.yml")
    args = parser.parse_args()
    run_cfg, cur_cfg = load_all(args.run)
    run_pipeline(run_cfg, cur_cfg)