"""Minimal local web app: pick a checkpoint, run it deterministically in the
numpy BaseDroneEnv, and watch the resulting flight as an mp4 in the browser.
Wraps app.guidance.record_run -- see that module for the actual rollout/
rendering logic.

Usage:
    uv run uvicorn app.guidance.serve_run:app --port 8000
Then open http://localhost:8000 -- local/dev tool only, no auth, synchronous
(one request blocks until the episode is rolled out and the mp4 is encoded,
usually a few seconds).
"""
import glob
import os
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.guidance.record_run import (
    CHECKPOINT_SEARCH_DIRS, N_EVAL_SCENARIOS, DEFAULT_SWITCH_DIST, record_checkpoint, evaluate_checkpoint,
    rank_checkpoints, record_checkpoint_two_phase, evaluate_two_phase,
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
    return sorted(set(paths), key=os.path.getmtime, reverse=True)


class RecordRequest(BaseModel):
    checkpoint: str | None = None
    distance_low: float = 3.0
    distance_high: float = 10.0
    seed: int | None = None
    # > 0 if the checkpoint was trained with --residual-scale (base_training_isaac.py PID-residual
    # mode) -- must match the value training used, or the composed action is wrong (see
    # record_run.RESIDUAL_GAINS_PATH's comment: feeding a residual checkpoint's raw output alone
    # means "hover, don't steer", so every episode times out regardless of checkpoint quality).
    residual_scale: float = 0.0


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
    residual_scale: float = 0.0


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
                               distance_high=req.distance_high, seed=req.seed,
                               residual_scale=req.residual_scale)
    return {"video_url": f"/recordings/{out_name}", "checkpoint": checkpoint, **stats}


@app.post("/api/evaluate")
def api_evaluate(req: EvaluateRequest):
    checkpoints = list_checkpoints()
    checkpoint = req.checkpoint or (checkpoints[0] if checkpoints else None)
    if checkpoint is None or not os.path.exists(checkpoint):
        raise HTTPException(404, f"checkpoint not found: {checkpoint}")

    result = evaluate_checkpoint(checkpoint, n_scenarios=req.n_scenarios, distance_low=req.distance_low,
                                  distance_high=req.distance_high, seed=req.seed,
                                  residual_scale=req.residual_scale)
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
                               distance_high=req.distance_high, seed=req.seed, top_n=req.top_n,
                               residual_scale=req.residual_scale)
    summary_name = f"{uuid.uuid4().hex}.txt"
    with open(os.path.join(RECORDINGS_DIR, summary_name), "w") as f:
        f.write(result["summary_text"])
    return {"summary_url": f"/recordings/{summary_name}", **result}


@app.post("/api/two_phase/record")
async def api_two_phase_record(
    file: UploadFile = File(...),
    distance: float = Form(30.0),
    switch_dist: float = Form(DEFAULT_SWITCH_DIST),
    seed: int | None = Form(None),
    residual_scale: float = Form(0.0),
):
    ckpt_path = _save_upload(file)
    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(RECORDINGS_DIR, out_name)
    stats = record_checkpoint_two_phase(ckpt_path, out_path, distance=distance, seed=seed,
                                         switch_dist=switch_dist, residual_scale=residual_scale)
    return {"video_url": f"/recordings/{out_name}", "checkpoint": file.filename, **stats}


@app.post("/api/two_phase/evaluate")
async def api_two_phase_evaluate(
    file: UploadFile = File(...),
    distance: float = Form(30.0),
    switch_dist: float = Form(DEFAULT_SWITCH_DIST),
    seed: int | None = Form(None),
    n_scenarios: int = Form(N_EVAL_SCENARIOS),
    residual_scale: float = Form(0.0),
):
    ckpt_path = _save_upload(file)
    result = evaluate_two_phase(ckpt_path, n_scenarios=n_scenarios, distance=distance, seed=seed,
                                 switch_dist=switch_dist, residual_scale=residual_scale)
    summary_name = f"{uuid.uuid4().hex}.txt"
    with open(os.path.join(RECORDINGS_DIR, summary_name), "w") as f:
        f.write(result["summary_text"])
    return {"summary_url": f"/recordings/{summary_name}", "checkpoint": file.filename, **result}


HTML_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Drone Policy Viewer</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  select, input, button { font-size: 14px; padding: 6px 8px; margin: 4px 0; }
  button { cursor: pointer; }
  label { display: block; margin-top: 10px; font-size: 13px; color: #444; }
  #status { margin: 10px 0; font-size: 13px; color: #666; }
  video { width: 100%; margin-top: 12px; background: #000; }
  #stats { font-size: 13px; color: #333; white-space: pre-wrap; margin-top: 8px; }
  .row { display: flex; gap: 12px; }
  .row > div { flex: 1; }
  #evalSummary, #rankSummary, #tpLog, #tpEvalSummary { font-family: monospace; font-size: 12px; white-space: pre;
                 overflow-x: auto; background: #f6f6f6; padding: 10px; margin-top: 8px; max-height: 480px;
                 overflow-y: auto; }
  hr { margin: 28px 0; border: none; border-top: 1px solid #ddd; }
</style>
</head>
<body>
<h2>Drone Policy Viewer</h2>
<label>Checkpoint</label>
<select id="checkpoint"></select>

<div class="row">
  <div>
    <label>Distance low (m)</label>
    <input id="distLow" type="number" value="3" step="0.5" style="width:100%">
  </div>
  <div>
    <label>Distance high (m)</label>
    <input id="distHigh" type="number" value="10" step="0.5" style="width:100%">
  </div>
  <div>
    <label>Seed (optional)</label>
    <input id="seed" type="number" style="width:100%">
  </div>
  <div>
    <label>Residual scale (0 = direct policy)</label>
    <input id="residualScale" type="number" value="0" step="0.05" min="0" style="width:100%">
  </div>
</div>
<div id="residualHint" style="font-size:12px;color:#a60;margin-top:2px;"></div>

<button id="runBtn" onclick="runRecord()">Record &amp; Play</button>
<button id="evalBtn" onclick="runEvaluate()">Run Full Evaluation (75 scenarios)</button>
<div id="status"></div>
<video id="player" controls autoplay loop hidden></video>
<div id="stats"></div>
<pre id="evalSummary" hidden></pre>

<hr>
<h2>Two-Phase Test (upload weights)</h2>
<p style="font-size:13px;color:#555;">
  PID always flies the midcourse leg alone -- gains solved analytically for the exact target
  distance (not a pre-tuned nearest-neighbor snap -- extrapolates past the old 250m ladder
  ceiling fine). At the switch distance, control engages: with Residual scale at 0 (a standalone
  policy) the checkpoint takes over fully; with Residual scale &gt; 0, PID keeps flying and the
  checkpoint's correction is added on top instead (required for a checkpoint trained with
  --residual-scale). Either way, set Switch distance at or inside the checkpoint's own trained
  distance range -- hand-off works best where the checkpoint has actually flown before. Log
  lines are wall-clock [HH:MM:SS.mmm], not simulated flight time. Each run's target is placed
  at exactly the distance below, in a random direction.
</p>
<label>Checkpoint weights (.pt)</label>
<input id="tpFile" type="file" accept=".pt" onchange="updateTpResidualHint()">

<div class="row">
  <div>
    <label>Target distance (m)</label>
    <input id="tpDistance" type="number" value="30" step="1" style="width:100%" oninput="updateBudgetHint()">
  </div>
  <div>
    <label>Switch distance (m)</label>
    <input id="tpSwitchDist" type="number" value="20" step="1" style="width:100%">
  </div>
  <div>
    <label>Seed (optional)</label>
    <input id="tpSeed" type="number" style="width:100%">
  </div>
  <div>
    <label>Residual scale (0 = standalone policy)</label>
    <input id="tpResidualScale" type="number" value="0" step="0.05" min="0" style="width:100%"
           oninput="updateTpResidualHint()">
  </div>
</div>
<div id="tpBudgetHint" style="font-size:12px;color:#a60;margin-top:2px;"></div>
<div id="tpResidualHint" style="font-size:12px;color:#a60;margin-top:2px;"></div>

<button id="tpRunBtn" onclick="runTwoPhaseRecord()">Record &amp; Play (two-phase)</button>
<div id="tpStatus"></div>
<video id="tpPlayer" controls autoplay loop hidden></video>
<div id="tpStats"></div>
<pre id="tpLog" hidden></pre>

<div class="row" style="margin-top:16px;">
  <div>
    <label>Test runs</label>
    <input id="tpNRuns" type="number" value="75" step="1" style="width:100%">
  </div>
</div>
<button id="tpEvalBtn" onclick="runTwoPhaseEvaluate()">Run Test Runs (two-phase, no video)</button>
<div id="tpEvalStatus"></div>
<pre id="tpEvalSummary" hidden></pre>

<h3>Rank checkpoints in a directory</h3>
<label>Checkpoint directory (e.g. runs/base_training_isaac_3_10)</label>
<input id="rankDir" type="text" style="width:100%" placeholder="runs/base_training_isaac_3_10">
<label>Top N</label>
<input id="topN" type="number" value="5" style="width:80px">
<button id="rankBtn" onclick="runRank()">Rank Checkpoints</button>
<div id="rankStatus"></div>
<pre id="rankSummary" hidden></pre>

<script>
async function loadCheckpoints() {
  const res = await fetch('/api/checkpoints');
  const data = await res.json();
  const sel = document.getElementById('checkpoint');
  sel.innerHTML = '';
  for (const cp of data.checkpoints) {
    const opt = document.createElement('option');
    opt.value = cp;
    opt.textContent = cp;
    sel.appendChild(opt);
  }
  if (data.checkpoints.length === 0) {
    sel.innerHTML = '<option value="">(no .pt checkpoints found under runs/ or app/control/)</option>';
  }
  sel.onchange = updateResidualHint;
  updateResidualHint();
}

function residualScale() {
  return parseFloat(document.getElementById('residualScale').value) || 0;
}

// Best-effort nudge, not a guarantee: a checkpoint from a run trained with --residual-scale > 0
// evaluates as "hover, don't steer" if residual_scale is left at 0 here (see record_run.py's
// RESIDUAL_GAINS_PATH comment) -- path naming is the only signal this page has, so this is a hint,
// not a check; always confirm against the run's own logged residual_scale (mlflow/metrics.csv).
function updateResidualHint() {
  const cp = (document.getElementById('checkpoint').value || '').toLowerCase();
  const hint = document.getElementById('residualHint');
  hint.textContent = (residualScale() === 0 && /res(idual)?[_-]?v?\d*[\\/]/.test(cp))
    ? "This checkpoint's path looks like a residual-mode run -- if it was trained with --residual-scale, set the same value above or every episode will just hover."
    : '';
}

async function runRecord() {
  const btn = document.getElementById('runBtn');
  const status = document.getElementById('status');
  const player = document.getElementById('player');
  const stats = document.getElementById('stats');
  btn.disabled = true;
  status.textContent = 'Running episode + encoding mp4... this can take up to a minute.';
  stats.textContent = '';
  player.hidden = true;

  const body = {
    checkpoint: document.getElementById('checkpoint').value || null,
    distance_low: parseFloat(document.getElementById('distLow').value),
    distance_high: parseFloat(document.getElementById('distHigh').value),
    seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
    residual_scale: residualScale(),
  };

  try {
    const res = await fetch('/api/record', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    status.textContent = 'Done.';
    player.src = data.video_url;
    player.hidden = false;
    stats.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function runEvaluate() {
  const evalBtn = document.getElementById('evalBtn');
  const status = document.getElementById('status');
  const summary = document.getElementById('evalSummary');
  evalBtn.disabled = true;
  status.textContent = 'Running 75 scenarios... this can take a minute or two.';
  summary.hidden = true;

  const body = {
    checkpoint: document.getElementById('checkpoint').value || null,
    distance_low: parseFloat(document.getElementById('distLow').value),
    distance_high: parseFloat(document.getElementById('distHigh').value),
    seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
    residual_scale: residualScale(),
  };

  try {
    const res = await fetch('/api/evaluate', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    status.textContent = 'Done. Summary saved to ' + data.summary_url;
    summary.textContent = data.summary_text;
    summary.hidden = false;
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    evalBtn.disabled = false;
  }
}

async function runRank() {
  const rankBtn = document.getElementById('rankBtn');
  const status = document.getElementById('rankStatus');
  const summary = document.getElementById('rankSummary');
  rankBtn.disabled = true;
  status.textContent = 'Evaluating every checkpoint in the directory... this can take a while.';
  summary.hidden = true;

  const body = {
    dir: document.getElementById('rankDir').value,
    top_n: parseInt(document.getElementById('topN').value) || 5,
    distance_low: parseFloat(document.getElementById('distLow').value),
    distance_high: parseFloat(document.getElementById('distHigh').value),
    seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
    residual_scale: residualScale(),
  };

  try {
    const res = await fetch('/api/rank', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    status.textContent = 'Done. Summary saved to ' + data.summary_url;
    summary.textContent = data.summary_text;
    summary.hidden = false;
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    rankBtn.disabled = false;
  }
}

// Must match record_run.MAX_TEST_DISTANCE -- distances up to this get their full,
// correctly-sized step budget (steps_for_dist); beyond it, the server clamps the
// budget and the episode may time out short of the target. Purely a client-side
// heads-up -- the server enforces the real clamp regardless of this value.
const TP_BUDGET_CAP_DIST = 5000;
// Below the hard cap, distances past this still take real wall-clock minutes per
// episode (steps_for_dist scales linearly with distance, physics steps at a fixed
// rate) -- just a "this will take a while" nudge, nothing is clamped yet.
const TP_SLOW_HINT_DIST = 500;

function tpResidualScale() {
  return parseFloat(document.getElementById('tpResidualScale').value) || 0;
}

// Same best-effort nudge as updateResidualHint (see its comment) -- here based on the
// uploaded FILE NAME (no server-side path to check), since /api/two_phase/* takes an
// upload, not a picked checkpoint.
function updateTpResidualHint() {
  const fileInput = document.getElementById('tpFile');
  const name = (fileInput.files.length ? fileInput.files[0].name : '').toLowerCase();
  const hint = document.getElementById('tpResidualHint');
  if (tpResidualScale() > 0) {
    hint.textContent = 'Residual scale > 0: PID flies alone until Switch distance, then PID + ' +
      'correction engage together. Set Switch distance at or inside the distance range this ' +
      'checkpoint was actually trained on, or hand-off happens somewhere the correction never saw.';
  } else if (/res(idual)?[_-]?v?\d*/.test(name)) {
    hint.textContent = "This checkpoint's filename looks like a residual-mode run -- if it was trained " +
      "with --residual-scale, set the same Residual scale above, or the RL phase's raw output alone " +
      "means 'hover, don't steer' and every run will time out regardless of checkpoint quality.";
  } else {
    hint.textContent = '';
  }
}

function updateBudgetHint() {
  const dist = parseFloat(document.getElementById('tpDistance').value) || 0;
  const hint = document.getElementById('tpBudgetHint');
  if (dist > TP_BUDGET_CAP_DIST) {
    hint.textContent = `Distances past ${TP_BUDGET_CAP_DIST}m get a clamped (shorter-than-ideal) step ` +
      `budget -- the episode may time out before reaching the target. See MAX_TEST_DISTANCE in record_run.py.`;
  } else if (dist > TP_SLOW_HINT_DIST) {
    hint.textContent = `At ${dist}m this episode gets a full settle-time budget, but expect real ` +
      `wall-clock minutes to run (longer per Test Run when batching many).`;
  } else {
    hint.textContent = '';
  }
}

function tpFormData(extra) {
  const fileInput = document.getElementById('tpFile');
  if (!fileInput.files.length) throw new Error('Upload a .pt checkpoint first.');
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('distance', document.getElementById('tpDistance').value);
  fd.append('switch_dist', document.getElementById('tpSwitchDist').value);
  fd.append('residual_scale', tpResidualScale());
  const seed = document.getElementById('tpSeed').value;
  if (seed) fd.append('seed', seed);
  for (const [k, v] of Object.entries(extra || {})) fd.append(k, v);
  return fd;
}

async function runTwoPhaseRecord() {
  const btn = document.getElementById('tpRunBtn');
  const status = document.getElementById('tpStatus');
  const player = document.getElementById('tpPlayer');
  const stats = document.getElementById('tpStats');
  const log = document.getElementById('tpLog');
  btn.disabled = true;
  stats.textContent = '';
  log.hidden = true;
  player.hidden = true;

  try {
    status.textContent = 'Running episode (PID guide -> RL takeover) + encoding mp4...';
    const fd = tpFormData();
    const res = await fetch('/api/two_phase/record', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    status.textContent = 'Done.';
    player.src = data.video_url;
    player.hidden = false;
    stats.textContent = JSON.stringify(
      { checkpoint: data.checkpoint, distance: data.distance, reason: data.reason, steps: data.steps,
        final_dist: data.final_dist, switch_dist: data.switch_dist, residual_scale: data.residual_scale }, null, 2);
    log.textContent = (data.log || []).join('\\n');
    log.hidden = false;
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function runTwoPhaseEvaluate() {
  const btn = document.getElementById('tpEvalBtn');
  const status = document.getElementById('tpEvalStatus');
  const summary = document.getElementById('tpEvalSummary');
  const nRuns = document.getElementById('tpNRuns').value || 75;
  btn.disabled = true;
  summary.hidden = true;

  try {
    status.textContent = `Running ${nRuns} test runs (PID guide -> RL takeover)... this can take a while.`;
    const fd = tpFormData({ n_scenarios: nRuns });
    const res = await fetch('/api/two_phase/evaluate', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    status.textContent = 'Done. Summary saved to ' + data.summary_url;
    summary.textContent = data.summary_text + '\\n\\n--- timestamped log ---\\n' + (data.log || []).join('\\n');
    summary.hidden = false;
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

loadCheckpoints();
updateBudgetHint();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
