# PyTorch Debug Env

A complete OpenEnv environment for the Meta PyTorch Hackathon where an AI agent investigates and diagnoses broken PyTorch training jobs. The design is optimized for the hackathon rubric: real-world utility, deterministic grading, multi-step interaction, and strong score variance on hard tasks.

## Why this project

The Round 1 problem statement requires a real-world task, full OpenEnv compliance, at least 3 graded tasks, a meaningful reward function, Dockerized deployment, Hugging Face Space deployment, and a reproducible `inference.py` baseline. OpenEnv itself is built around typed models, `step()`, `reset()`, `state()`, and environment packaging, which makes a multi-step debugging environment a natural fit.

This project targets **PyTorch training failure diagnosis**, which is highly relevant to the hackathon theme and gives a strong real-world utility story. The environment is designed as a true agent-training loop rather than a one-shot benchmark: the agent reveals evidence over multiple steps, updates a hypothesis, and is rewarded for improving its diagnosis over time.

***

## Core concept

Each episode provides a synthetic but realistic PyTorch training repository containing code, logs, curves, and resource profiles. The agent can inspect files and artifacts, maintain a current hypothesis, and finally commit to a diagnosis.

The environment state changes as the agent investigates, so rewards are dense and trajectory-aware instead of purely terminal. That matches the hackathon's requirement for partial progress signals and meaningful reward shaping.

***

## High-level architecture

```text
pytorch-debug-env/
├── openenv.yaml
├── inference.py
├── Dockerfile
├── requirements.txt
├── README.md
├── src/
│   └── pytorch_debug_env/
│       ├── __init__.py
│       ├── models.py
│       ├── bug_library.py
│       ├── scenario_generator.py
│       ├── graders.py
│       ├── reward.py
│       ├── environment.py
│       └── server.py
├── tests/
│   ├── test_reward.py
│   ├── test_graders.py
│   └── test_environment.py
└── scenarios/
    └── seeds.json
```

### Module responsibilities

| File | Responsibility |
|------|----------------|
| `models.py` | Pydantic/OpenEnv typed observation, action, reward, and state models |
| `bug_library.py` | Bug templates, bug metadata, categories, ground-truth structure |
| `scenario_generator.py` | Builds synthetic repos, logs, curves, memory profiles, and ground truth |
| `graders.py` | Deterministic scoring for easy, medium, and hard tasks |
| `reward.py` | Hypothesis-quality and step reward computation |
| `environment.py` | Main OpenEnv implementation: `reset()`, `step()`, `state()` |
| `server.py` | FastAPI/OpenEnv app entrypoint for container runtime |
| `inference.py` | Required baseline script using OpenAI client and structured logs |

***

## Environment mechanics

### Episode flow

1. `reset()` creates or loads one scenario.
2. The initial observation reveals only part of the repo and a partial artifact window.
3. On each `step()`, the agent submits:
   - a current hypothesis,
   - optionally an investigation action,
   - optionally a final diagnosis commitment.
4. The environment reveals more evidence, updates state, computes reward, and returns feedback.
5. Episode ends when max steps are reached or the agent commits a diagnosis.

This multi-step design is important because OpenEnv is explicitly structured around iterative interaction rather than single-shot evaluation.

***

## Task design

The environment includes three tasks with deterministic graders and increasing difficulty, satisfying the hackathon requirement for at least 3 tasks with grader-based scores in the 0.0-1.0 range.

### Task 1 — Single-file bug detection

**Goal:** Identify a straightforward training bug from obvious signals.

Examples:
- missing `optimizer.zero_grad()`
- wrong loss function
- tensor shape mismatch

**Expected agent behavior:** infer bug from early loss behavior or traceback and commit quickly.

### Task 2 — Multi-file root cause analysis

**Goal:** Correlate evidence across two or more files.

Examples:
- data leakage in preprocessing and split logic
- scheduler misuse caused by config + train loop mismatch
- N+1 data loading bottleneck

**Expected agent behavior:** inspect the right files and refine the hypothesis.

### Task 3 — Silent failure diagnosis

**Goal:** Diagnose a subtle failure with no easy traceback and with red herrings.

Examples:
- class-level dataset cache causing memory leak
- mixed precision overflow in custom layer
- distributed sync bug in multi-GPU training

**Expected agent behavior:** investigate systematically and avoid overfitting to obvious but wrong signals.