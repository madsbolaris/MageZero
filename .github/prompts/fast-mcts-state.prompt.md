# FastMctsState Engine Rewrite — Agent Prompt

## Context

You are optimizing the MCTS engine for [MageZero](https://github.com/madsbolaris/mage), a reinforcement learning framework for Magic: The Gathering built on the XMage game engine. The Java repo is at `/Users/mabolan/repos/github/madsbolaris/mage`, branch `mcts-perf`.

## The Problem

Every MCTS simulation calls `validateState()` which performs a **full deep copy of the XMage GameState** (36+ fields, including battlefield with 50-100 permanents, player hands/libraries/graveyards, spell stack, continuous effects, triggers, watchers, and card state). This copy is the dominant cost — roughly **70-80% of per-simulation CPU time**.

Current throughput: **~40 sims/sec** on M1 Mac. Target: **200-400 sims/sec** (5-10x).

The copy happens because:
1. There is one shared `rootGame` object per MCTS tree
2. For each new node, `validateState()` copies `parent.state`, loads it into `rootGame`, rebuilds effects, then replays all actions since the last stable state using `PlayerScript` sequences
3. After replaying, it captures the new state for nodes at PRIORITY decision points (stable states)
4. Micro-decision nodes (CHOOSE_TARGET, CHOOSE_USE, etc.) reuse the parent's state

## Key Files

All paths relative to repo root `/Users/mabolan/repos/github/madsbolaris/mage`:

### MCTS Core
- `Mage.Server.Plugins/Mage.Player.AIMCTS/src/mage/player/ai/MCTSNode.java` — base MCTS node class
  - `validateState()` (line ~300) — THE HOT PATH. Copies GameState, resets rootGame, replays actions, encodes state
  - `resetRootGame(GameState)` (line ~375) — loads state into shared game, calls `applyEffects()`
  - `populateActionScripts()` (line ~395) — walks parent chain to build replay scripts
  - `expand()` (line ~545) — creates child nodes after validation
- `Mage.Server.Plugins/Mage.Player.AI.RL/src/mage/player/ai/MCTSNode2.java` — async RL node (extends MCTSNode)
  - `evaluate()` (line ~48) — sends state to neural network for inference
- `Mage.Server.Plugins/Mage.Player.AI.RL/src/mage/player/ai/ComputerPlayerMCTS2.java` — top-level MCTS player
  - `applyMCTS()` (line ~145) — main search loop calling `validateState()` per expansion
  - `getNextAction()` (line ~244) — root setup

### Game Engine
- `Mage/src/main/java/mage/game/GameState.java` — the state object being copied
  - Copy constructor (line ~108) — 36+ field deep copy, ~50 lines
  - Most expensive fields: `battlefield` (20-25%), `players` (15-20%), `stack` (10-15%), `effects` (8-12%), `triggers` (8-10%)
- `Mage.Server.Plugins/Mage.Player.AIMCTS/src/mage/player/ai/MCTSPlayer.java` — scripted action replay
  - `priority()` — polls from `actionScript.prioritySequence` and calls `activateAbility()`

### State Encoding
- `Mage.Server.Plugins/Mage.Player.AI/src/main/java/mage/player/ai/encoder/StateEncoder.java`
  - `processState()` (line ~602) — hashes game state into sparse binary feature vector (~200 active features)

## Current Architecture (What Happens Per Sim)

```
applyMCTS loop:
  1. select() — walk tree via PUCT to a leaf node
  2. validateState() on the leaf:
     a. populateActionScripts() — walk parent chain, collect all actions since last PRIORITY
     b. parent.state.copy() — FULL DEEP COPY of GameState (36+ fields)
     c. resetRootGame(copiedState) — load into shared Game, call applyEffects()
     d. game.resume() — replay actions via PlayerScript until next decision
     e. Capture: terminal?, winner?, playerId, actionType, stateVector, prefixScripts
     f. If PRIORITY: this.state = rootGame.getState() (save for future children)
     g. If micro-decision: this.state = parent.state (reuse parent's)
  3. Dedup check via transposition index
  4. evaluate() — async NN inference
  5. expand() — create child nodes
  6. backpropagate()
```

## Design Constraints

1. **Correctness is paramount.** The new fast path must produce identical legal actions, state vectors, and game outcomes as the current code. Any divergence means wrong training data → broken RL.

2. **XMage is not our code.** The core game engine (`GameImpl`, `GameState`, `Player`, `Permanent`, etc.) is upstream XMage code. We cannot modify it extensively. Our changes must live in the MCTS plugin layer (`MCTSNode`, `MCTSNode2`, `ComputerPlayerMCTS2`).

3. **PRIORITY states are the checkpoints.** The current design already recognizes this — only PRIORITY nodes save their own `state`. Micro-decision nodes reuse parent's state and replay from there. This is the right foundation.

4. **The Game.resume() replay is necessary** because XMage effects, triggers, and state-based actions can only be evaluated by running the game engine. We can't skip it — but we can avoid re-running from scratch every time.

5. **The feature vector (stateVector) must remain identical.** It's the input to the neural network. Any change to features breaks trained models.

## Proposed Approach: Incremental State Management

### Tier 1: Avoid Redundant Copies (Moderate Effort)

Instead of `parent.state.copy()` for every node, recognize that **siblings share the same parent state**. The copy is needed because `resetRootGame()` mutates the game — but if we process siblings in order and restore state between them, we can avoid redundant copies.

Key insight: after `validateState()` runs for node A (child of P), the rootGame is in A's post-replay state. To validate node B (also child of P), we currently copy P.state again and replay B's script. Instead:
- Keep P.state loaded in rootGame between sibling validations
- Only copy+reload when moving to a different parent

This requires changing the MCTS loop to validate siblings together rather than one-at-a-time as leaves are discovered. Trade-off: changes the expansion order, which may affect search dynamics.

### Tier 2: Checkpoint + Delta Replay (High Effort)

For a chain of micro-decisions (e.g., PRIORITY → CHOOSE_TARGET → CHOOSE_TARGET → CHOOSE_USE), the current code copies the PRIORITY state and replays all actions each time. Instead:

- Save the rootGame state as a "checkpoint" after each validated node
- For the next node in the chain, restore the checkpoint and only replay the incremental action
- This turns O(depth) replay into O(1) per node

Implementation:
```java
class FastCheckpoint {
    GameState checkpoint;          // saved at last validated node
    PlayerScript incrementalScript; // only the new action(s) since checkpoint
}
```

### Tier 3: Copy-on-Write GameState (Very High Effort)

Replace the flat deep copy with structural sharing:
- Battlefield, stack, exile, etc. use persistent/immutable data structures
- `copy()` returns a shallow wrapper that copies-on-write only the modified parts
- Most MCTS simulations only modify 1-2 permanents per step

This requires deep changes to XMage's data structures (breaks constraint #2 somewhat) but would make `copy()` near-instantaneous.

## Build & Test

```bash
# Build
cd /Users/mabolan/repos/github/madsbolaris/mage
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
mvn -pl Mage.MageZero -am install -DskipTests -Dmaven.javadoc.skip=true -q

# Copy jars to MageZero
cp Mage.Server.Plugins/Mage.Player.AIMCTS/target/mage-player-ai-mcts.jar \
   /Users/mabolan/repos/github/WillWroble/MageZero/xmage/lib/mage-player-ai-mcts-1.4.58.jar
cp Mage.Server.Plugins/Mage.Player.AI.RL/target/mage-player-ai-rl.jar \
   /Users/mabolan/repos/github/WillWroble/MageZero/xmage/lib/mage-player-ai-rl-1.4.58.jar
cp Mage.Server.Plugins/Mage.Player.AI/target/mage-player-ai.jar \
   /Users/mabolan/repos/github/WillWroble/MageZero/xmage/lib/mage-player-ai-1.4.58.jar

# Smoke test (5 games, ~5 min)
cd /Users/mabolan/repos/github/WillWroble/MageZero
source .venv/bin/activate
export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"
mz benchmark --run configs/run.baylen-smoke.yml

# Full benchmark (15 games, ~15 min)
mz benchmark
```

## Success Criteria

1. Smoke test passes: 5/5 games complete, 0 failed
2. Benchmark shows measurable sims/sec improvement (>20% to justify complexity)
3. Win rates and game outcomes are statistically consistent with baseline
4. No new crashes or `IllegalStateException` errors

## What NOT to Do

- Do NOT delete `data/`, `models/`, or `runs/` directories — they contain hours of training work
- Do NOT modify the neural network architecture or feature encoding — only the MCTS search engine
- Do NOT change the MCTS algorithm (PUCT, backprop, etc.) — only the state management
- Do NOT remove the old `validateState()` code path until the new one is verified via shadow validation
- Do NOT add external dependencies without checking if they're already in the Maven tree

## Related Issues

- madsbolaris/mage#26 (C48): FastMctsState
- madsbolaris/mage#27 (C49): Checkpoint only at stable priority windows
- madsbolaris/mage#28 (C50): Apply/undo for common MTG actions
- madsbolaris/mage#5 (J5): GameState copy-on-write
- madsbolaris/mage#6 (J6): Skip redundant parent.state.copy() in validateState
