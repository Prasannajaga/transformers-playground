# AGENTS.md — Strict Structural Governance

This repository follows a strict separation between:

- src/            → importable core library (pure Python modules)
- experiments/    → project-specific execution entry points
- scripts/        → global automation & operational utilities
- configs/        → configuration files only
- notebooks/      → analysis & playground only
- data/           → raw datasets (no code)
- templates/      → metadata templates only

Any violation of these boundaries introduces architectural debt.

------------------------------------------------------------

1. DIRECTORY RESPONSIBILITIES

src/

- Contains reusable library code only.
- No CLI logic.
- No Dockerfiles.
- No shell scripts.
- No notebooks.
- No dataset files.
- No experiment-specific training scripts.

experiments/

- Contains experiment-specific entrypoints.
- May import from src/.
- Must not redefine core layers or architectures.
- Must not duplicate training loops.

scripts/

- Global automation only.
- No model definitions.
- No training loop definitions.
- No architectural logic.

configs/

- YAML configuration files only.
- No executable logic.
- No hardcoded training inside Python configs.

notebooks/

- Exploration only.
- Must not contain production logic.

data/

- Raw data only.
- Never imported directly inside src/.

templates/

- Static templates only.
- No execution logic.

------------------------------------------------------------

1. STRUCTURAL RULES (MANDATORY)

- Do not create new top-level folders.
- Do not create nested project structures inside src/.
- Do not introduce wrapper layers.
- Do not introduce Base/Factory/Manager abstractions.
- Do not duplicate functionality across experiments.
- Do not move training logic into src/utils/.
- Do not create separate training scripts per model variant.

All model variants must be config-driven.

------------------------------------------------------------

1. TRAINING LOGIC POLICY

- Training loop logic must NOT live in experiments/.
- experiments/*/train.py must only:
  - load config
  - build model
  - call shared training logic

If training behavior differs (pretrain, finetune, distill),
handle it via configuration flags or strategy functions
inside src/, not via new scripts.

------------------------------------------------------------

1. LAYER BOUNDARIES

src/layers/

- Primitive building blocks only.
- No training code.
- No cloud logic.

src/architectures/

- Model composition only.
- No training loops.
- No dataset loading.

src/optimization/

- Pure algorithmic transformations (e.g., pruning).
- No deployment logic.

src/losses/

- Loss functions only.

src/utils/

- Small pure helpers only.
- No training orchestration.
- No experiment logic.

------------------------------------------------------------

1. INTEGRATION RULES

HuggingFace, cloud, and external APIs must:

- Live in src/utils/ as thin adapters.
- Not wrap internal architecture.
- Not redefine model behavior.
- Not introduce abstraction layers.

No "wrapper classes" allowed.

------------------------------------------------------------

1. PERFORMANCE DISCIPLINE

- Avoid unnecessary tensor copies.
- Avoid .clone() unless required.
- Avoid device transfers inside loops.
- Use torch.no_grad() for inference.
- Never load full datasets into memory.
- Prefer streaming or batched loading.

------------------------------------------------------------

1. SCOPE CONTROL

Agents must:

- Implement only the requested change.
- Not refactor unrelated files.
- Not reorganize directories.
- Not add documentation noise.
- Not add test files.
- Not introduce speculative improvements.

Strict minimalism is required.

------------------------------------------------------------
