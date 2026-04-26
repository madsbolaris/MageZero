"""
cli.py — `mz` command entry point.

Commands:
  mz train                          full curriculum pipeline (auto-resume)
  mz batch [--config FILE]          single JVM launch via game.yml
  mz play  --deck X [--version N]   host a local AI player (stub)
  mz import <file>                  auto-detects .dck or .mz (.txt stubbed)
  mz export --deck X --version N    pack model into a .mz bundle
  mz benchmark --run FILE           measure generation throughput
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from magezero.util.config import load_all
from magezero import runner


# ─── train ───────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    run_cfg, cur_cfg = load_all(args.run)
    runner.run_pipeline(run_cfg, cur_cfg, base_game_yml=args.game)


# ─── batch ───────────────────────────────────────────────────

def cmd_batch(args: argparse.Namespace) -> None:
    runner.launch_jvm(args.config)


# ─── play ────────────────────────────────────────────────────

def cmd_play(args: argparse.Namespace) -> None:
    import subprocess
    from pathlib import Path

    config = Path(args.config).resolve()
    if not config.exists():
        sys.exit(f"config not found: {config}")

    deck = args.deck
    version = args.version
    if version is None:
        version = runner.latest_version(deck)
        if version is None:
            sys.exit(f"no trained model found for {deck}")

    if not runner.has_checkpoint(deck, version):
        sys.exit(f"no checkpoint at models/{deck}/ver{version}/model.pt.gz")

    # start inference server
    print(f"[play] starting inference server for {deck} v{version}")
    server = runner.start_server(deck, version, runner.PRIMARY_PORT, Path("."))

    try:
        # launch XMage server
        script = "xmage\\mz-xmage-play.bat" if sys.platform == "win32" else "xmage/mz-xmage-play.sh"
        cmd = ["cmd", "/c", script, str(config)] if sys.platform == "win32" else [script, str(config)]
        subprocess.run(cmd, check=True)
    finally:
        runner.stop_server(server)


# ─── import ──────────────────────────────────────────────────

def cmd_import(args: argparse.Namespace) -> None:
    src = Path(args.file)
    if not src.exists():
        sys.exit(f"file not found: {src}")

    suffix = src.suffix.lower()
    if suffix == ".dck":
        dst = Path("xmage/decks") / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"✓ imported deck → {dst}")

    elif suffix == ".mz":
        with zipfile.ZipFile(src) as zf:
            meta = json.loads(zf.read("metadata.json"))
            deck = meta["deck"]
            version = meta["version"]
            dst = Path("models") / deck / f"ver{version}"
            dst.mkdir(parents=True, exist_ok=True)
            zf.extract("model.pt.gz", dst)
            zf.extract("ignore.roar", dst)
        print(f"✓ imported model → {dst}")

    elif suffix == ".txt":
        sys.exit("`.txt` deck conversion not yet wired up. "
                 "Convert manually to .dck for now.")

    else:
        sys.exit(f"unknown file type: {suffix} (expected .dck, .mz, or .txt)")


# ─── export ──────────────────────────────────────────────────

def cmd_export(args: argparse.Namespace) -> None:
    src = Path("models") / args.deck / f"ver{args.version}"
    if not src.exists():
        sys.exit(f"model not found: {src}")

    model_file = src / "model.pt.gz"
    ignore_file = src / "ignore.roar"
    if not model_file.exists() or not ignore_file.exists():
        sys.exit(f"missing model.pt.gz or ignore.roar in {src}")

    out_dir = Path("exports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.deck}_v{args.version}.mz"

    metadata = {"deck": args.deck, "version": args.version}
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_file, "model.pt.gz")
        zf.write(ignore_file, "ignore.roar")
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    print(f"✓ exported → {out_path}")


# ─── benchmark ───────────────────────────────────────────────

def cmd_benchmark(args: argparse.Namespace) -> None:
    run_cfg, cur_cfg = load_all(args.run)
    print(f"[benchmark] deck={run_cfg.deck} v{run_cfg.version} "
          f"opponent={run_cfg.opponents[0].deck} games={run_cfg.games_per_gen}")
    try:
        report = runner.run_benchmark(run_cfg, cur_cfg, base_game_yml=args.game)
    except RuntimeError as e:
        sys.exit(str(e))

    # Pretty-print the report
    print("\n" + "=" * 60)
    print("  BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Wall time:           {report['wall_time_sec']}s")
    print(f"  Games/hour:          {report['games_per_hour']}")
    print(f"  Games completed:     {report['jvm']['games_successful']} "
          f"(failed: {report['jvm']['games_failed']})")
    print(f"  Win rate:            {report['jvm']['win_rate_pct']}%")
    print(f"  MCTS sims/sec (avg): {report['jvm']['mcts_sims_per_sec_mean']}")
    print(f"  MCTS sims/sec (end): {report['jvm']['mcts_sims_per_sec_final']}")
    if report.get("server"):
        srv = report["server"]
        print(f"  Inferences/sec:      {srv.get('inferences_per_sec', 'N/A')}")
        print(f"  Batches/sec:         {srv.get('batches_per_sec', 'N/A')}")
        print(f"  Latency p50:         {srv.get('latency_p50_ms', 'N/A')}ms")
        print(f"  Latency p95:         {srv.get('latency_p95_ms', 'N/A')}ms")
        print(f"  Latency p99:         {srv.get('latency_p99_ms', 'N/A')}ms")
        dist = srv.get("batch_size_distribution", {})
        if dist:
            print(f"  Batch size dist:     {dict(list(dist.items())[:10])}")
    else:
        print("  Inference:           offline (no server)")
    print(f"  Mode:                {'offline' if report['config']['offline'] else 'online'}")
    print("=" * 60)

    # Save timestamped report to benchmarks/ history folder
    bench_dir = Path("benchmarks")
    bench_dir.mkdir(exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_name = f"{ts}_{run_cfg.deck}_v{run_cfg.version}.json"
    out_path = bench_dir / out_name
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved: {out_path}")

    # Compare against previous best in the folder
    prev_reports = sorted(
        [f for f in bench_dir.glob("*.json") if f != out_path],
        key=lambda f: f.name,
    )
    if prev_reports:
        prev = json.loads(prev_reports[-1].read_text())
        print(f"\n  vs previous ({prev_reports[-1].name}):")
        _compare("Games/hour", prev.get("games_per_hour", 0), report["games_per_hour"])
        _compare("MCTS sims/sec", prev.get("jvm", {}).get("mcts_sims_per_sec_mean", 0),
                 report["jvm"]["mcts_sims_per_sec_mean"])
        prev_srv = prev.get("server") or {}
        cur_srv = report.get("server") or {}
        if prev_srv and cur_srv:
            _compare("Inferences/sec", prev_srv.get("inferences_per_sec", 0),
                     cur_srv.get("inferences_per_sec", 0))
            _compare("Latency p95 (ms)", prev_srv.get("latency_p95_ms", 0),
                     cur_srv.get("latency_p95_ms", 0), lower_is_better=True)
    else:
        print("\n  (first benchmark — no previous to compare against)")


def _compare(label: str, old: float, new: float, lower_is_better: bool = False) -> None:
    if old == 0:
        print(f"    {label:25s}  {new:>10}  (no baseline)")
        return
    delta_pct = (new - old) / old * 100
    if lower_is_better:
        arrow = "▼" if delta_pct < 0 else "▲" if delta_pct > 0 else "="
        color = "better" if delta_pct < 0 else "worse" if delta_pct > 0 else "same"
    else:
        arrow = "▲" if delta_pct > 0 else "▼" if delta_pct < 0 else "="
        color = "better" if delta_pct > 0 else "worse" if delta_pct < 0 else "same"
    print(f"    {label:25s}  {old:>10} → {new:>10}  {arrow} {abs(delta_pct):+.1f}% ({color})")


# ─── main ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="mz")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="full curriculum pipeline")
    p_train.add_argument("--run", default="configs/run.yml")
    p_train.add_argument("--game", default="configs/game.yml")
    p_train.set_defaults(func=cmd_train)

    p_batch = sub.add_parser("batch", help="single JVM launch")
    p_batch.add_argument("--config", default="configs/game.yml")
    p_batch.set_defaults(func=cmd_batch)

    p_play = sub.add_parser("play", help="host a local AI player")
    p_play.add_argument("--deck", required=True)
    p_play.add_argument("--version", type=int, default=None)
    p_play.add_argument("--config", default="configs/game.yml")
    p_play.set_defaults(func=cmd_play)

    p_import = sub.add_parser("import", help="import .dck or .mz file")
    p_import.add_argument("file")
    p_import.set_defaults(func=cmd_import)

    p_export = sub.add_parser("export", help="export model as .mz bundle")
    p_export.add_argument("--deck", required=True)
    p_export.add_argument("--version", type=int, required=True)
    p_export.set_defaults(func=cmd_export)

    p_bench = sub.add_parser("benchmark", help="measure generation throughput")
    p_bench.add_argument("--run", default="configs/run.baylen-smoke.yml")
    p_bench.add_argument("--game", default="configs/game.yml")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()