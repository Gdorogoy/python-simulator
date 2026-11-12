"""Local web viewer server: fly a checkpoint in the numpy simulator and watch it in 3D, or score it in a batch.

    uv run uvicorn app.guidance.serve_run:app --port 8000   # then open http://localhost:8000

What is in this file, top to bottom:
    1. paths and settings
    2. checkpoint helpers (list the .pt files, save an upload, read a run's config)
    3. the form fields each tab sends (FastAPI reads them from the submitted form)
    4. the endpoints: /api/checkpoints, /api/checkpoint_info, /api/two_phase/*, /api/acro/*
    5. the web page itself (the React files in app/guidance/web/)

The heavy work (flying the drone) lives in record_run.py; this file only turns form fields into function calls.
The endpoints are plain `def`, so FastAPI runs each one in a worker thread and a long episode does not freeze the server.
"""
import glob
import json
import os
import shutil
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from app.guidance.record_run import (
    CHECKPOINT_SEARCH_DIRS, N_EVAL_SCENARIOS, DEFAULT_SWITCH_DIST,
    evaluate_acro, trajectory_acro,
    record_checkpoint_two_phase, evaluate_two_phase, trajectory_two_phase,
)

# ----------------------------------------------------------------------------------------------------------
# 1. paths and settings
# ----------------------------------------------------------------------------------------------------------
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")   # index.html, styles.css, js/
RECORDINGS_DIR = "recordings"     # mp4 videos and text summaries made by the endpoints
UPLOAD_DIR = "uploaded_weights"   # .pt files uploaded from the page
ALLOWED_ACTION_REPEATS = (1, 2, 4, 8)   # physics steps (240 Hz) each acro action is held; 4 = the 60 Hz Isaac trains at

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Drone Policy Viewer")
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")


# ----------------------------------------------------------------------------------------------------------
# 2. checkpoint helpers
# ----------------------------------------------------------------------------------------------------------
def list_checkpoints():
    """Every .pt under the search folders. best.pt files come first, then the newest files."""
    paths = []
    for folder in CHECKPOINT_SEARCH_DIRS:
        paths += glob.glob(os.path.join(folder, "**", "*.pt"), recursive=True)
    # latest_full.pt also holds the critic and optimizer, so it cannot be loaded for inference
    paths = [p for p in set(paths) if os.path.basename(p) != "latest_full.pt"]
    return sorted(paths, key=lambda p: (os.path.basename(p) != "best.pt", -os.path.getmtime(p)))


def is_listed_checkpoint(path):
    """True if `path` is one of the files list_checkpoints() returns (stops the page asking for arbitrary files)."""
    listed = {os.path.realpath(p) for p in list_checkpoints()}
    return os.path.realpath(path) in listed


def save_upload(file: UploadFile):
    """Saves an uploaded checkpoint under UPLOAD_DIR with a unique prefix (so two uploads named model.pt never collide)."""
    safe_name = os.path.basename(file.filename or "checkpoint.pt")
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{safe_name}")
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return path


def save_summary(summary_text):
    """Writes a batch summary to recordings/ and returns the URL the page can open."""
    name = f"{uuid.uuid4().hex}.txt"
    with open(os.path.join(RECORDINGS_DIR, name), "w") as f:
        f.write(summary_text)
    return f"/recordings/{name}"


def read_run_config(checkpoint_path):
    """The run_config.json the trainer wrote next to the checkpoint, or None if there is none."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "run_config.json")
    try:
        with open(config_path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------------------------------------
# 3. the form fields each tab sends
# ----------------------------------------------------------------------------------------------------------
class CheckpointChoice:
    """Which checkpoint to use: an uploaded file, or the path of a listed one."""

    def __init__(self, file: UploadFile | None = File(None), checkpoint: str | None = Form(None)):
        self.file = file
        self.picked = checkpoint

    def resolve(self):
        """Returns (path on disk, name to show). Unlisted paths are rejected."""
        if self.file is not None and self.file.filename:
            return save_upload(self.file), self.file.filename
        if self.picked:
            if not is_listed_checkpoint(self.picked):
                raise HTTPException(400, f"not a listed checkpoint: {self.picked}")
            return self.picked, self.picked
        raise HTTPException(400, "give either an uploaded .pt file or a listed checkpoint path")


class TwoPhaseSettings:
    """The Angle tab's scenario. `distance` is how far the drone starts from the target and `switch_dist` where the
    model takes over from the PID/carrot controller. With handoff off the model flies from the start by itself."""

    def __init__(
        self,
        distance: float = Form(30.0),
        switch_dist: float = Form(DEFAULT_SWITCH_DIST),
        seed: int | None = Form(None),
        cruise_speed: float = Form(18.0),
        handoff_speed: float | None = Form(None),
        recentre_z: float | None = Form(None),
        handoff: bool = Form(True),
        lateral_frac: float = Form(0.0),
        tilt_deg: float = Form(0.0),
        hit_radius: float | None = Form(None),
    ):
        if hit_radius is not None and hit_radius <= 0:
            raise HTTPException(400, "hit_radius must be above 0 m")
        self.hit_radius = hit_radius                 # None = the viewer env's default zone (rewards.HIT_THRESHOLD)
        self.distance = distance
        self.seed = seed
        self.recentre_z = recentre_z
        self.lateral_frac = lateral_frac
        self.tilt_deg = tilt_deg
        if handoff:
            self.switch_dist = switch_dist
            self.cruise_speed = cruise_speed or None     # 0 means "the old slow PID", which the functions call None
            self.handoff_speed = handoff_speed
        else:
            # "just the model": put the switch point past the spawn point so the controller never flies and nothing is snapped
            self.switch_dist = distance + 1.0
            self.cruise_speed = None
            self.handoff_speed = None

    def as_arguments(self):
        """Keyword arguments shared by record / trajectory / evaluate in record_run.py."""
        return dict(
            distance=self.distance, seed=self.seed, switch_dist=self.switch_dist, cruise_speed=self.cruise_speed,
            handoff_speed=self.handoff_speed, recentre_z=self.recentre_z, lateral_frac=self.lateral_frac, tilt_deg=self.tilt_deg,
            hit_radius=self.hit_radius,
        )


class ThreatSettings:
    """The Angle tab's chasing drones: `threats` of them, real drones flown by the limit-LOS controller (straight at the drone at
    `threat_speed`, the only limit). `threat_spawn` "uniform" starts each at a random point `threat_spawn_min`..`threat_spread` m from
    the base, "base" starts them all at the base with `threat_buffer` m clearance. Each hovers until the model has taken over and the
    drone has covered a fraction U(`threat_release_lo`, `threat_release_hi`) of the distance it had left at the handoff (the trainer's
    --threat-release-after). `threat_radius` is the collision sphere of the training run (drawn; the viewer sim has no contact, a catch
    is the reward's intercept radius). 0 threats = none. The checkpoint must have been built with that many threat slots
    (observation 23 + 7 * slots, plus 2 if it was trained with the zone timer)."""

    def __init__(
        self,
        threats: int = Form(0),
        threat_speed: float = Form(50.0),
        threat_speed_min: float | None = Form(None),
        threat_spread: float = Form(30.0),
        threat_spawn_min: float = Form(15.0),
        threat_spawn: str = Form("uniform"),
        threat_buffer: float = Form(0.5),
        threat_radius: float | None = Form(None),
        threat_release_lo: float = Form(0.33),
        threat_release_hi: float = Form(0.99),
        threat_max_tilt_deg: float = Form(90.0),
        threat_w_head_on: float = Form(1.0),
        threat_w_tail: float = Form(1.0),
        threat_reaction_s: float = Form(1.5),
        threat_tail_lo: float = Form(15.0),
        threat_tail_hi: float = Form(40.0),
        threat_jitter: float = Form(2.0),
    ):
        if threats < 0:
            raise HTTPException(400, "threats must be 0 or more")
        if threat_spawn not in ("uniform", "base", "geometry"):
            raise HTTPException(400, "threat_spawn must be 'uniform', 'base' or 'geometry'")
        weights = (threat_w_head_on, threat_w_tail)
        if min(weights) < 0 or sum(weights) <= 0:
            raise HTTPException(400, "geometry weights must be >= 0 and not all zero")
        if threat_reaction_s <= 0 or threat_jitter < 0 or not 0 <= threat_tail_lo <= threat_tail_hi:
            raise HTTPException(400, "geometry spawn: reaction time > 0, jitter >= 0, tail gap 0 <= lo <= hi")
        self.geometry = dict(geometry_weights=weights, reaction_s=threat_reaction_s,
                             tail_gap=(threat_tail_lo, threat_tail_hi), jitter=threat_jitter)
        if not 0.0 <= threat_release_lo <= threat_release_hi < 1.0:
            raise HTTPException(400, "release after needs 0 <= LO <= HI < 1")
        self.active, self.speed, self.speed_min, self.spread = threats, threat_speed, threat_speed_min, threat_spread
        self.spawn_min, self.spawn, self.buffer, self.radius = threat_spawn_min, threat_spawn, threat_buffer, threat_radius
        self.release_lo, self.release_hi, self.max_tilt_deg = threat_release_lo, threat_release_hi, threat_max_tilt_deg

    def as_arguments(self):
        """Keyword argument for trajectory / evaluate / record_two_phase in record_run.py. The trainer's "covered fraction" p becomes the
        env's "fraction still left" 1 - p."""
        return dict(threats=dict(active=self.active, speed=self.speed, speed_min=self.speed_min, spread=self.spread,
                                 spawn_min=self.spawn_min, spawn=self.spawn, buffer=self.buffer, radius=self.radius,
                                 release_frac_range=(1.0 - self.release_hi, 1.0 - self.release_lo), max_tilt_deg=self.max_tilt_deg,
                                 **self.geometry))


class AcroSettings:
    """The Acro tab's scenario: the same fields as the Angle tab plus the acro-only ones.
    With handoff on, `distance` is the spawn distance and `switch_dist` the handoff point (where the policy takes over).
    With handoff off the policy flies from `distance` away."""

    def __init__(
        self,
        distance: float = Form(40.0),
        switch_dist: float = Form(4.0),
        seed: int | None = Form(None),
        cruise_speed: float = Form(8.0),
        handoff_speed: float | None = Form(None),
        handoff: bool = Form(True),
        lateral_frac: float = Form(0.3),
        tilt_deg: float = Form(15.0),
        tilt_limit_deg: float = Form(120.0),
        action_repeat: int = Form(4),
        start: str = Form("auto"),
        hit_radius: float | None = Form(None),
        runner_speed: float | None = Form(None),
        runner_speed_min: float | None = Form(None),
        runner_escape_dist: float | None = Form(None),
    ):
        if action_repeat not in ALLOWED_ACTION_REPEATS:
            raise HTTPException(400, "action_repeat must be 1, 2, 4 or 8")
        if hit_radius is not None and hit_radius <= 0:
            raise HTTPException(400, "hit_radius must be above 0 m")
        for name, value in (("runner_speed", runner_speed), ("runner_speed_min", runner_speed_min), ("runner_escape_dist", runner_escape_dist)):
            if value is not None and value <= 0:
                raise HTTPException(400, f"{name} must be above 0")
        self.hit_radius = hit_radius                 # None = the radius the run was trained with
        # moving target (a --runner run only; ignored otherwise): None = the run's own value
        self.runner = dict(speed=runner_speed, speed_min=runner_speed_min, escape_dist=runner_escape_dist)
        self.seed = seed
        self.cruise_speed = cruise_speed
        self.handoff_speed = handoff_speed
        self.lateral_frac = lateral_frac
        self.tilt_deg = tilt_deg
        self.tilt_limit_deg = tilt_limit_deg
        self.action_repeat = action_repeat
        self.start = start
        if handoff:
            self.policy_distance = switch_dist
            self.spawn_distance = distance
        else:
            self.policy_distance = distance
            self.spawn_distance = None

    def as_arguments(self):
        """Keyword arguments shared by trajectory_acro and evaluate_acro (the spawn distance is added by each endpoint)."""
        return dict(
            seed=self.seed, tilt_limit_deg=self.tilt_limit_deg, action_repeat=self.action_repeat, start=self.start,
            max_speed=self.cruise_speed, handoff_speed=self.handoff_speed, lateral_frac=self.lateral_frac, tilt_deg=self.tilt_deg,
            hit_radius=self.hit_radius, runner=self.runner,
        )


def _release_after(config):
    """[lo, hi]: the fraction of the handoff distance the drone has covered when a threat is released, as the run was trained.
    Old runs saved the fixed "fraction still left" threat_release_frac instead (0.7 = after covering 30%); None = not recorded."""
    after, left = config.get("threat_release_after"), config.get("threat_release_frac")
    if left is not None:
        return [round(1.0 - float(left), 4)] * 2
    return [float(after[0]), float(after[1])] if after else None


@app.get("/api/checkpoints")
def api_checkpoints():
    return {"checkpoints": list_checkpoints()}


@app.get("/api/checkpoint_info")
def api_checkpoint_info(path: str):
    """What the run behind a listed checkpoint was trained with, so the page can pre-set its fields.
    Fields the run did not record come back as null."""
    if not is_listed_checkpoint(path):
        raise HTTPException(400, f"not a listed checkpoint: {path}")
    config = read_run_config(path)
    if config is None:
        return {"has_config": False}
    trained_with_handoff = bool(config.get("handoff"))
    has_spawn_leg = config.get("spawn_dist_low") is not None
    return {
        "has_config": True,
        "distance_low": config.get("distance_low"),
        "distance_high": config.get("distance_high"),
        "spawn_dist_low": config.get("spawn_dist_low"),
        "spawn_dist_high": config.get("spawn_dist_high"),
        "max_speed": config.get("max_speed") if trained_with_handoff else None,
        "lateral_frac": config.get("lateral_frac") if trained_with_handoff else None,
        "handoff_tilt_deg": config.get("handoff_tilt_deg") if trained_with_handoff else None,
        "is_acro": "tilt_limit" in config,
        # chasing drones the run trained against (num_threats = observation slots; active defaults to all of them)
        "num_threats": config.get("num_threats") or 0,
        "active_threats": (config.get("num_threats") or 0) if config.get("active_threats") is None else config.get("active_threats"),
        "threat_speed": config.get("threat_speed"),
        "threat_spread": config.get("threat_spread"),
        "threat_speed_min": config.get("threat_speed_min"),
        "threat_spawn_min": config.get("threat_spawn_min"),
        "threat_spawn": config.get("threat_spawn"),
        "threat_buffer": config.get("threat_buffer"),
        "threat_radius": config.get("threat_radius"),
        "threat_release_after": _release_after(config),
        "threat_geometry_weights": config.get("threat_geometry_weights"),
        "threat_reaction_s": config.get("threat_reaction_s"),
        "threat_tail_gap": config.get("threat_tail_gap"),
        "threat_jitter": config.get("threat_jitter"),
        "los_max_tilt_deg": config.get("los_max_tilt_deg"),
        "obs_position_mode": config.get("obs_position_mode") or "full",
        "obs_layout": config.get("obs_layout") or "v1",
        # acro moving target (--runner)
        "runner": bool(config.get("runner")),
        "runner_speed": config.get("runner_speed"),
        "runner_speed_min": config.get("runner_speed_min"),
        "runner_escape_dist": config.get("runner_escape_dist"),
        "hit_radius": (config.get("acro_reward") or {}).get("hit_threshold"),
        # a run with a spawn leg re-centres the handoff point to x=y=0 unless --no-recenter was given
        "recentred": bool(has_spawn_leg and config.get("recenter", True)),
    }


# ---- Angle tab ----
@app.post("/api/two_phase/trajectory")
def api_two_phase_trajectory(
    choice: CheckpointChoice = Depends(), settings: TwoPhaseSettings = Depends(), chasers: ThreatSettings = Depends(),
):
    """One angle flight as per-frame pose data for the 3D player."""
    path, name = choice.resolve()
    try:
        stats = trajectory_two_phase(path, **settings.as_arguments(), **chasers.as_arguments())
    except ValueError as error:          # e.g. more chasers than the checkpoint has threat slots
        raise HTTPException(400, str(error))
    return {"checkpoint": name, **stats}


@app.post("/api/two_phase/evaluate")
def api_two_phase_evaluate(
    choice: CheckpointChoice = Depends(),
    settings: TwoPhaseSettings = Depends(),
    chasers: ThreatSettings = Depends(),
    n_scenarios: int = Form(N_EVAL_SCENARIOS),
):
    """Many angle flights at one distance, scored without drawing anything."""
    path, name = choice.resolve()
    try:
        result = evaluate_two_phase(path, n_scenarios=n_scenarios, **settings.as_arguments(), **chasers.as_arguments())
    except ValueError as error:
        raise HTTPException(400, str(error))
    return {**result, "checkpoint": name, "summary_url": save_summary(result["summary_text"])}


@app.post("/api/two_phase/record")
def api_two_phase_record(
    choice: CheckpointChoice = Depends(), settings: TwoPhaseSettings = Depends(), chasers: ThreatSettings = Depends(),
):
    """Like /trajectory, but renders an mp4 video (slower). The page does not use this; it is kept for scripts."""
    path, name = choice.resolve()
    video_name = f"{uuid.uuid4().hex}.mp4"
    try:
        stats = record_checkpoint_two_phase(path, os.path.join(RECORDINGS_DIR, video_name), **settings.as_arguments(),
                                            **chasers.as_arguments())
    except ValueError as error:
        raise HTTPException(400, str(error))
    plot_path = stats.pop("plot_path", None)
    plot_url = f"/recordings/{os.path.basename(plot_path)}" if plot_path else None
    return {"video_url": f"/recordings/{video_name}", "plot_url": plot_url, "checkpoint": name, **stats}


# ---- Acro tab ----
@app.post("/api/acro/trajectory")
def api_acro_trajectory(choice: CheckpointChoice = Depends(), settings: AcroSettings = Depends()):
    """One acro flight as per-frame pose data for the 3D player."""
    path, name = choice.resolve()
    try:
        stats = trajectory_acro(path, distance=settings.policy_distance, spawn_dist=settings.spawn_distance, **settings.as_arguments())
    except ValueError as error:          # e.g. the checkpoint is not an acro checkpoint
        raise HTTPException(400, str(error))
    return {"checkpoint": name, **stats}


@app.post("/api/acro/evaluate")
def api_acro_evaluate(
    choice: CheckpointChoice = Depends(),
    settings: AcroSettings = Depends(),
    n_scenarios: int = Form(N_EVAL_SCENARIOS),
):
    """Many acro flights at one distance, scored like the trainer's isaac_eval."""
    path, name = choice.resolve()
    try:
        result = evaluate_acro(
            path, n_scenarios=n_scenarios,
            distance_low=settings.policy_distance, distance_high=settings.policy_distance,          # one value, not a range
            spawn_dist_low=settings.spawn_distance, spawn_dist_high=settings.spawn_distance,
            **settings.as_arguments(),
        )
    except ValueError as error:
        raise HTTPException(400, str(error))
    return {**result, "checkpoint": name, "summary_url": save_summary(result["summary_text"])}


# ----------------------------------------------------------------------------------------------------------
# 5. the web page
# ----------------------------------------------------------------------------------------------------------
# Mounted last so it only answers URLs no endpoint above matched. Files are read from disk on every request,
# so editing app/guidance/web/ shows up on a browser refresh. "/" serves web/index.html.
@app.middleware("http")
async def no_cache_web(request, call_next):
    """Browsers cache ES modules hard; without this a refresh can keep running old js/ files after an edit."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
