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
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.guidance.record_run import (
    CHECKPOINT_SEARCH_DIRS, N_EVAL_SCENARIOS, record_checkpoint, evaluate_checkpoint, rank_checkpoints,
)

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

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


class EvaluateRequest(RecordRequest):
    n_scenarios: int = N_EVAL_SCENARIOS


class RankRequest(BaseModel):
    dir: str
    distance_low: float = 3.0
    distance_high: float = 10.0
    seed: int | None = None
    n_scenarios: int = N_EVAL_SCENARIOS
    top_n: int = 5


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
  #evalSummary, #rankSummary { font-family: monospace; font-size: 12px; white-space: pre; overflow-x: auto;
                 background: #f6f6f6; padding: 10px; margin-top: 8px; max-height: 480px; overflow-y: auto; }
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
</div>

<button id="runBtn" onclick="runRecord()">Record &amp; Play</button>
<button id="evalBtn" onclick="runEvaluate()">Run Full Evaluation (75 scenarios)</button>
<div id="status"></div>
<video id="player" controls autoplay loop hidden></video>
<div id="stats"></div>
<pre id="evalSummary" hidden></pre>

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

loadCheckpoints();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
