# Copilot Instructions for MageZero

## CRITICAL: Never delete training artifacts

**NEVER run `rm -rf data/`, `rm -rf models/`, or `rm -rf runs/` unless the user explicitly asks you to delete them.** These directories contain hours/days of training work:

- `models/` — trained neural network checkpoints (hours of GPU time)
- `data/` — game data from self-play (hours of CPU time)  
- `runs/` — run history, logs, training curves

When starting a new training run, use `start_from_version` in the run config to build on existing checkpoints instead of bootstrapping from scratch. Only clean these directories for smoke tests when the user explicitly requests a fresh start.

## Training data is precious

- A single generation of 200 games × 5 opponents takes ~3 hours on M1 Mac
- A full 6-generation run takes 12-18 hours
- Deleting a checkpoint means redoing the offline bootstrap (~3 hours) before online training can begin
- Always prefer resuming or building on existing data over starting fresh

## Before launching training runs

1. Check if `models/<deck>/ver<N>/model.pt.gz` exists
2. If it does, set `start_from_version: <N>` in the run config to skip bootstrap
3. Only set `start_from_version: null` if no checkpoint exists or user explicitly wants a fresh start

## Project structure

- Python repo: `/Users/mabolan/repos/github/WillWroble/MageZero` (branch: macos-support, remote: mads → madsbolaris/MageZero)
- Java repo: `/Users/mabolan/repos/github/madsbolaris/mage` (branch: mcts-perf, remote: origin → madsbolaris/mage)
- Java build: `JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home mvn -pl Mage.MageZero -am install -DskipTests -Dmaven.javadoc.skip=true -q`
- Jar names in build output have no version suffix; MageZero expects `-1.4.58` suffix
- Python venv: `.venv/`, activate with `source .venv/bin/activate`
- Benchmark: `mz benchmark` (defaults to 15-game config)
