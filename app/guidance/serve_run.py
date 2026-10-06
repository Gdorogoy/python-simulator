"""Local web viewer: run a checkpoint (single or two-phase) in the numpy env and watch it in 3D / mp4.

    uv run uvicorn app.guidance.serve_run:app --port 8000   # then open http://localhost:8000
"""
import glob
import json
import os
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.guidance.record_run import (
    CHECKPOINT_SEARCH_DIRS, N_EVAL_SCENARIOS, DEFAULT_SWITCH_DIST, record_checkpoint, evaluate_checkpoint,
    rank_checkpoints, record_checkpoint_two_phase, evaluate_two_phase, trajectory_two_phase,
)

RECORDINGS_DIR = "recordings"
UPLOAD_DIR = "uploaded_weights"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Drone Policy Viewer")
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")


def list_checkpoints():
    paths = []
    for d in CHECKPOINT_SEARCH_DIRS:
        paths += glob.glob(os.path.join(d, "**", "*.pt"), recursive=True)
    # latest_full.pt holds critic + optimizer (not loadable for inference); best.pt first, then newest
    paths = [p for p in set(paths) if os.path.basename(p) != "latest_full.pt"]
    return sorted(paths, key=lambda p: (os.path.basename(p) != "best.pt", -os.path.getmtime(p)))


class RecordRequest(BaseModel):
    checkpoint: str | None = None
    distance_low: float = 3.0
    distance_high: float = 10.0
    seed: int | None = None


class EvaluateRequest(RecordRequest):
    n_scenarios: int = N_EVAL_SCENARIOS


def _save_upload(file: UploadFile) -> str:
    """Saves an uploaded checkpoint under UPLOAD_DIR with a unique prefix
    (so two uploads named model.pt never collide) and returns its path."""
    safe_name = os.path.basename(file.filename or "checkpoint.pt")
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{safe_name}")
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return path


class RankRequest(BaseModel):
    dir: str
    distance_low: float = 3.0
    distance_high: float = 10.0
    seed: int | None = None
    n_scenarios: int = N_EVAL_SCENARIOS
    top_n: int = 5


@app.get("/api/checkpoint_info")
def api_checkpoint_info(path: str):
    """What the run that produced a listed checkpoint was trained with (its run_config.json, written by
    rtl_train_isaac.py), so the page can pre-set the two-phase options. Unknown fields are simply absent."""
    allowed = {os.path.realpath(p) for p in list_checkpoints()}
    if os.path.realpath(path) not in allowed:
        raise HTTPException(400, f"not a listed checkpoint: {path}")
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(path)), "run_config.json")
    info = {"has_config": False}
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return info
    info.update(
        has_config=True,
        distance_low=cfg.get("distance_low"), distance_high=cfg.get("distance_high"),
        spawn_dist_low=cfg.get("spawn_dist_low"), spawn_dist_high=cfg.get("spawn_dist_high"),
        max_speed=cfg.get("max_speed") if cfg.get("handoff") else None,
        obs_position_mode=cfg.get("obs_position_mode") or "full",
        # a spawn-leg run re-centres the handoff point to x=y=0 unless --no-recenter was given
        recentred=bool(cfg.get("spawn_dist_low") is not None and cfg.get("recenter", True)),
    )
    return info


@app.get("/api/checkpoints")
def api_checkpoints():
    return {"checkpoints": list_checkpoints()}


@app.post("/api/record")
def api_record(req: RecordRequest):
    checkpoints = list_checkpoints()
    checkpoint = req.checkpoint or (checkpoints[0] if checkpoints else None)
    if checkpoint is None or not os.path.exists(checkpoint):
        raise HTTPException(404, f"checkpoint not found: {checkpoint}")

    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(RECORDINGS_DIR, out_name)
    stats = record_checkpoint(checkpoint, out_path, distance_low=req.distance_low,
                               distance_high=req.distance_high, seed=req.seed)
    return {"video_url": f"/recordings/{out_name}", "checkpoint": checkpoint, **stats}


@app.post("/api/evaluate")
def api_evaluate(req: EvaluateRequest):
    checkpoints = list_checkpoints()
    checkpoint = req.checkpoint or (checkpoints[0] if checkpoints else None)
    if checkpoint is None or not os.path.exists(checkpoint):
        raise HTTPException(404, f"checkpoint not found: {checkpoint}")

    result = evaluate_checkpoint(checkpoint, n_scenarios=req.n_scenarios, distance_low=req.distance_low,
                                  distance_high=req.distance_high, seed=req.seed)
    summary_name = f"{uuid.uuid4().hex}.txt"
    with open(os.path.join(RECORDINGS_DIR, summary_name), "w") as f:
        f.write(result["summary_text"])
    return {"summary_url": f"/recordings/{summary_name}", **result}


@app.post("/api/rank")
def api_rank(req: RankRequest):
    if not os.path.isdir(req.dir):
        raise HTTPException(404, f"directory not found: {req.dir}")
    checkpoints = sorted(glob.glob(os.path.join(req.dir, "*.pt")))
    if not checkpoints:
        raise HTTPException(404, f"no .pt files in {req.dir}")

    result = rank_checkpoints(checkpoints, n_scenarios=req.n_scenarios, distance_low=req.distance_low,
                               distance_high=req.distance_high, seed=req.seed, top_n=req.top_n)
    summary_name = f"{uuid.uuid4().hex}.txt"
    with open(os.path.join(RECORDINGS_DIR, summary_name), "w") as f:
        f.write(result["summary_text"])
    return {"summary_url": f"/recordings/{summary_name}", **result}


def _resolve_checkpoint(upload: UploadFile | None, picked: str | None) -> tuple[str, str]:
    """Resolve an uploaded .pt or a listed checkpoint path -> (path, name); unlisted paths are rejected."""
    if upload is not None and upload.filename:
        return _save_upload(upload), upload.filename
    if picked:
        allowed = {os.path.realpath(p) for p in list_checkpoints()}
        if os.path.realpath(picked) not in allowed:
            raise HTTPException(400, f"not a listed checkpoint: {picked}")
        return picked, picked
    raise HTTPException(400, "give either an uploaded .pt file or a listed checkpoint path")


@app.post("/api/two_phase/record")
async def api_two_phase_record(
    file: UploadFile | None = File(None),
    checkpoint: str | None = Form(None),
    distance: float = Form(30.0),
    switch_dist: float = Form(DEFAULT_SWITCH_DIST),
    seed: int | None = Form(None),
    cruise_speed: float = Form(18.0),
    handoff_speed: float | None = Form(None),
    recentre_z: float | None = Form(None),
):
    ckpt_path, name = _resolve_checkpoint(file, checkpoint)
    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(RECORDINGS_DIR, out_name)
    stats = record_checkpoint_two_phase(ckpt_path, out_path, distance=distance, seed=seed,
                                         switch_dist=switch_dist, cruise_speed=cruise_speed or None,
                                         handoff_speed=handoff_speed, recentre_z=recentre_z)
    plot_path = stats.pop("plot_path", None)
    plot_url = f"/recordings/{os.path.basename(plot_path)}" if plot_path else None
    return {"video_url": f"/recordings/{out_name}", "plot_url": plot_url, "checkpoint": name, **stats}


@app.post("/api/two_phase/trajectory")
async def api_two_phase_trajectory(
    file: UploadFile | None = File(None),
    checkpoint: str | None = Form(None),
    distance: float = Form(30.0),
    switch_dist: float = Form(DEFAULT_SWITCH_DIST),
    seed: int | None = Form(None),
    cruise_speed: float = Form(18.0),
    handoff_speed: float | None = Form(None),
    recentre_z: float | None = Form(None),
):
    """Same episode as /api/two_phase/record but returns the per-frame pose as JSON for the three.js player
    (no mp4 encoding, so it is faster)."""
    ckpt_path, name = _resolve_checkpoint(file, checkpoint)
    stats = trajectory_two_phase(ckpt_path, distance=distance, seed=seed, switch_dist=switch_dist,
                                  cruise_speed=cruise_speed or None, handoff_speed=handoff_speed, recentre_z=recentre_z)
    return {"checkpoint": name, **stats}


@app.post("/api/two_phase/evaluate")
async def api_two_phase_evaluate(
    file: UploadFile | None = File(None),
    checkpoint: str | None = Form(None),
    distance: float = Form(30.0),
    switch_dist: float = Form(DEFAULT_SWITCH_DIST),
    seed: int | None = Form(None),
    cruise_speed: float = Form(18.0),
    handoff_speed: float | None = Form(None),
    recentre_z: float | None = Form(None),
    n_scenarios: int = Form(N_EVAL_SCENARIOS),
):
    ckpt_path, name = _resolve_checkpoint(file, checkpoint)
    result = evaluate_two_phase(ckpt_path, n_scenarios=n_scenarios, distance=distance, seed=seed,
                                 switch_dist=switch_dist, cruise_speed=cruise_speed or None,
                                         handoff_speed=handoff_speed, recentre_z=recentre_z)
    summary_name = f"{uuid.uuid4().hex}.txt"
    with open(os.path.join(RECORDINGS_DIR, summary_name), "w") as f:
        f.write(result["summary_text"])
    return {"summary_url": f"/recordings/{summary_name}", **{**result, "checkpoint": name}}


# The page lives in viewer.html (next to this file) so it can be edited as a normal HTML file.
VIEWER_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(VIEWER_HTML, encoding="utf-8") as f:     # read per request: edits show up on refresh
        return f.read()
