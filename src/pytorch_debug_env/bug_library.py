# src/pytorch_debug_env/bug_library.py
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np


@dataclass
class BugTemplate:
    bug_type: str
    category: str
    difficulty: str
    primary_bug_file: str
    related_files: List[str]
    red_herring_file: Optional[str]
    fix_strategy: str
    line_range: List[int]
    description: str
    artifact_generator: Callable
    repo_mutator: Callable
    metadata: Dict = field(default_factory=dict)


BUG_CATEGORIES = {
    "shape_mismatch": "model",
    "missing_zero_grad": "optimization",
    "wrong_loss_function": "optimization",
    "learning_rate_too_high": "optimization",
    "gradient_explosion": "optimization",
    "memory_leak": "resource",
    "data_leakage": "data",
    "incorrect_normalization": "data",
    "distributed_sync_error": "distributed",
    "amp_overflow": "numerics",
}

# Realistic artifact generator
def dummy_artifact_generator(artifact_type: str, rng):
    if artifact_type == "loss_curve":
        t = np.arange(100)
        base = 2.3 * np.exp(-0.01 * t) + 0.15
        oscillation = 0.22 * np.sin(0.25 * t) * np.exp(-0.002 * t)
        return [
            {"step": int(i), "train_loss": float(base[i] + oscillation[i])}
            for i in range(100)
        ]
    elif artifact_type == "gpu_profile":
        t = np.arange(100)
        allocated = 2048 + 2.4 * t
        return [
            {"step": int(i), "allocated_mb": float(allocated[i])}
            for i in range(100)
        ]
    elif artifact_type == "training_log":
        return "Epoch 1, Step 0: loss 2.45\nEpoch 1, Step 1: loss 2.43\n"
    return []

def dummy_repo_mutator(repo_files, rng):
    return repo_files

BUG_TEMPLATES = [
    BugTemplate(
        bug_type="missing_zero_grad",
        category="optimization",
        difficulty="easy",
        primary_bug_file="train.py",
        related_files=[],
        red_herring_file="model/architecture.py",
        fix_strategy="Call optimizer.zero_grad() before loss.backward()",
        line_range=[10, 15],
        description="Missing zero grad",
        artifact_generator=dummy_artifact_generator,
        repo_mutator=dummy_repo_mutator,
    ),
    BugTemplate(
        bug_type="data_leakage",
        category="data",
        difficulty="medium",
        primary_bug_file="data/dataset.py",
        related_files=["data/preprocessing.py"],
        red_herring_file="train.py",
        fix_strategy="Ensure validation split is strictly separate from training",
        line_range=[20, 25],
        description="Data leakage",
        artifact_generator=dummy_artifact_generator,
        repo_mutator=dummy_repo_mutator,
    ),
    BugTemplate(
        bug_type="memory_leak",
        category="resource",
        difficulty="hard",
        primary_bug_file="data/dataset.py",
        related_files=["train.py"],
        red_herring_file="model/attention.py",
        fix_strategy="Avoid holding reference to tensors in class cache",
        line_range=[30, 35],
        description="Memory leak",
        artifact_generator=dummy_artifact_generator,
        repo_mutator=dummy_repo_mutator,
    )
]
