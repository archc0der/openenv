# src/pytorch_debug_env/server.py
from fastapi import FastAPI

from .environment import PyTorchDebugEnv
from .models import PyTorchDebugAction
from .scenario_generator import ScenarioGenerator
from .bug_library import BUG_TEMPLATES

app = FastAPI(title="PyTorch Debug Env")

env = PyTorchDebugEnv(generator=ScenarioGenerator(BUG_TEMPLATES))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(task_id: str = "easy"):
    obs = await env.reset(task_id=task_id)
    return {"observation": obs, "done": False}


@app.post("/step")
async def step(action: PyTorchDebugAction):
    return await env.step(action)


@app.get("/state")
async def state():
    return await env.state()
