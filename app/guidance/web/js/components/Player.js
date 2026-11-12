// The 3D replay player: the three.js view (player/PlayerScene.js) plus the play bar, camera buttons and options.
//
// How it fits together:
//   - React state holds what the buttons control (playing, speed, camera, checkboxes).
//   - The current replay time lives in a ref (timeRef), because it changes 60 times a second.
//   - One requestAnimationFrame loop advances the time and asks the scene to draw.
import { useEffect, useRef, useState } from 'react';
import { html } from '../html.js';
import { PlayerScene, SPEEDS, CHASER_COLORS } from '../player/PlayerScene.js';
import { Segmented } from './ui.js';
import { ForcePanel } from './ForcePanel.js';

const DEFAULT_OPTIONS = { trail: true, ghost: true, vel: true, guide: true, drops: true, switch: true, chasers: true, rotorVec: true, forceVec: true, arcs: true };
const OPTION_LABELS = [
  ['trail', 'trail'], ['ghost', 'full path'], ['vel', 'velocity'],
  ['guide', 'line to target'], ['drops', 'ground drop'], ['switch', 'switch sphere'], ['chasers', 'chasing drones'],
  ['rotorVec', 'rotor vectors'], ['forceVec', 'force vectors'], ['arcs', 'angle arcs'],
];
// options that only mean something when the run has the matching data
const needsChasers = ['chasers'], needsForces = ['rotorVec', 'forceVec', 'arcs'];
const EMPTY_HUD = { time: 0, step: 0, dist: null, speed: null, policyFlying: false, chaserDist: null, chasers: null, los: null, dyn: null };
// 0 deg = flying straight at the target, 90 = sideways, 180 = straight away
const losColor = (deg) => (deg < 30 ? '#4cd964' : deg < 90 ? '#ff9f43' : '#ff6b6b');
const clamp = (x, low, high) => Math.min(Math.max(x, low), high);

export function Player({ track, meta }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const stageRef = useRef(null);
  const sceneRef = useRef(null);
  const timeRef = useRef(0);                 // seconds into the replay

  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(false);
  const [cameraMode, setCameraMode] = useState('orbit');
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [hud, setHud] = useState(EMPTY_HUD);

  // The animation loop and the keyboard handler must see the newest values without being re-created.
  const latest = useRef({});
  latest.current = { playing, speed, loop, cameraMode, options };

  const duration = () => (sceneRef.current ? sceneRef.current.duration : 0);

  // ---- create the scene once, run the animation loop, follow size changes ----
  useEffect(() => {
    const scene = new PlayerScene(canvasRef.current, chartRef.current);
    sceneRef.current = scene;

    const stage = stageRef.current;
    const resizeObserver = new ResizeObserver(() => {
      if (!stage.clientWidth || !stage.clientHeight) return;      // hidden tab: nothing to size
      scene.resize(stage.clientWidth, stage.clientHeight);
      scene.drawChart(timeRef.current);
    });
    resizeObserver.observe(stage);
    const chartObserver = new ResizeObserver(() => scene.drawChart(timeRef.current));
    chartObserver.observe(chartRef.current);

    let lastTimestamp = performance.now();
    let frameId = 0;
    function onFrame(timestamp) {
      frameId = requestAnimationFrame(onFrame);
      const seconds = Math.min((timestamp - lastTimestamp) / 1000, 0.1);
      lastTimestamp = timestamp;
      if (!scene.run || !stage.clientWidth) return;
      const state = latest.current;
      if (state.playing) {
        timeRef.current += seconds * state.speed;
        if (timeRef.current >= scene.duration) {
          if (state.loop) {
            timeRef.current = 0;
          } else {
            timeRef.current = scene.duration;
            setPlaying(false);
          }
        }
      }
      const info = scene.draw(timeRef.current, seconds, state.options, state.cameraMode);
      setHud({ time: timeRef.current, step: info.step, dist: info.dist, speed: info.speed, policyFlying: info.policyFlying, chaserDist: info.chaserDist, chasers: info.chasers, los: info.los, dyn: info.dyn });
    }
    frameId = requestAnimationFrame(onFrame);

    return () => {
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      chartObserver.disconnect();
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  // ---- a new run arrived: load it and reset the controls ----
  useEffect(() => {
    const autoSpeed = sceneRef.current.load(track, meta);
    timeRef.current = 0;
    setPlaying(false);
    setSpeed(autoSpeed);
    setCameraMode('orbit');
  }, [track, meta]);

  // ---- actions ----
  function seek(seconds) {
    timeRef.current = clamp(seconds, 0, duration());
  }
  function togglePlay() {
    if (!sceneRef.current.run) return;
    if (!playing && timeRef.current >= duration() - 1e-6) timeRef.current = 0;     // replay from the start
    setPlaying(!playing);
  }
  function stepFrames(count) {
    setPlaying(false);
    if (sceneRef.current.run) seek(timeRef.current + count * sceneRef.current.run.dt);
  }
  function jumpToEnd() {
    setPlaying(false);
    seek(duration());
  }
  function chooseCamera(mode) {
    if (mode === 'chase') sceneRef.current.startChase(timeRef.current);
    setCameraMode(mode);
  }
  function showView(kind) {
    setCameraMode('orbit');
    sceneRef.current.showView(kind);
  }
  function changeSpeed(step) {
    const index = clamp(SPEEDS.indexOf(speed) + step, 0, SPEEDS.length - 1);
    setSpeed(SPEEDS[index]);
  }
  function seekFromPointer(event) {
    const box = chartRef.current.getBoundingClientRect();
    seek(duration() * clamp((event.clientX - box.left) / box.width, 0, 1));
  }
  function toggleOption(name) {
    setOptions((previous) => ({ ...previous, [name]: !previous[name] }));
  }

  // ---- keyboard shortcuts (only while this player is visible and you are not typing in a text box) ----
  function onKey(event) {
    const scene = sceneRef.current;
    if (!scene || !scene.run || stageRef.current.offsetParent === null) return;
    const tag = (event.target.tagName || '').toLowerCase();
    const typing = tag === 'textarea' || tag === 'select' || (tag === 'input' && event.target.type !== 'range' && event.target.type !== 'checkbox');
    if (typing) return;
    const jump = event.shiftKey ? 10 : 1;
    const mode = cameraMode;
    const key = event.key;
    if (key === ' ') togglePlay();
    else if (key === 'ArrowLeft') stepFrames(-jump);
    else if (key === 'ArrowRight') stepFrames(jump);
    else if (key === 'Home') seek(0);
    else if (key === 'End') jumpToEnd();
    else if (key === 'f' || key === 'F') chooseCamera(mode === 'follow' ? 'orbit' : 'follow');
    else if (key === 'c' || key === 'C') chooseCamera(mode === 'chase' ? 'orbit' : 'chase');
    else if (key === 'r' || key === 'R') showView('fit');
    else if (key === '+' || key === '=' || key === ']') changeSpeed(1);
    else if (key === '-' || key === '[') changeSpeed(-1);
    else return;
    event.preventDefault();
  }
  // The listener is added once; it always calls the newest onKey (which sees the newest state).
  const keyHandler = useRef(onKey);
  keyHandler.current = onKey;
  useEffect(() => {
    const listener = (event) => keyHandler.current(event);
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, []);

  const total = duration();
  const finished = total > 0 && hud.time >= total - 1e-6;
  // Label the non-policy frames by what actually flew them. A run with no handoff (every frame phase 1) is the policy
  // from start to finish, so it never gets a controller label. The name of the scripted controller comes from the page
  // (meta.controller: "PID" on the angle tab, "controller" on the acro tab).
  const phases = track && track.phase;
  const allPolicy = Array.isArray(phases) && phases.length > 0 && phases.every((p) => p === 1);
  const showPolicy = hud.policyFlying || allPolicy;
  const controllerName = (meta && meta.controller) || 'controller';
  return html`
    <div className="player">
      <div className="pv-main">
      <div className="pv-stage" ref=${stageRef}>
        <canvas className="pv-canvas" ref=${canvasRef} onDoubleClick=${togglePlay}></canvas>
        <div className="pv-hud">
          <div>
            <b style=${{ color: showPolicy ? '#ff9f43' : '#7aa7ff' }}>${showPolicy ? 'RL policy' : controllerName}</b>
            ${' '}<span>${finished ? '- ' + meta.reason : ''}</span>
          </div>
          <div>t <b>${hud.time.toFixed(1)}</b> s ${'  '}step <b>${hud.step}</b></div>
          <div>dist <b>${hud.dist == null ? '-' : hud.dist.toFixed(2)}</b> m ${'  '}speed <b>${hud.speed == null ? '-' : hud.speed.toFixed(1)}</b> m/s</div>
          ${hud.los && html`<div title="angle between the velocity and the line to the target; reward = los_k * cos(angle), 0 below the min speed">
            LOS <b style=${{ color: losColor(hud.los.deg) }}>${hud.los.deg.toFixed(1)}</b>° ${'  '}reward <b>${hud.los.reward.toFixed(3)}</b>/step ${'  '}total <b>${hud.los.sum.toFixed(1)}</b>
            ${hud.los.k != null && html`<span style=${{ opacity: 0.6 }}> (k ${hud.los.k})</span>`}
          </div>`}
          ${hud.chaserDist != null && html`<div style=${{ color: '#ff6b6b' }}>nearest chaser <b>${hud.chaserDist.toFixed(2)}</b> m</div>`}
          ${hud.chasers && hud.chasers.map((c, k) => html`<div key=${k} title="chaser LOS = angle between the chaser's velocity and its line to the drone (arc at the chaser); 0 = flying straight at the drone">
            <span style=${{ color: '#' + CHASER_COLORS[k % CHASER_COLORS.length].toString(16).padStart(6, '0') }}>●</span> chaser ${k + 1}
            ${'  '}<b>${c.dist.toFixed(1)}</b> m ${'  '}speed <b>${c.speed == null ? '-' : c.speed.toFixed(1)}</b> m/s
            ${'  '}LOS <b style=${{ color: c.los == null ? undefined : losColor(c.los) }}>${c.los == null ? '-' : c.los.toFixed(1)}</b>°</div>`)}
        </div>
        <div className="pv-help">drag: orbit · right-drag: pan · wheel: zoom · space: play/pause · ←/→: step · F: follow</div>
      </div>
      ${track.dyn && html`<${ForcePanel} dyn=${hud.dyn} meta=${track.dyn_meta} />`}
      </div>

      <canvas className="pv-chart" ref=${chartRef} title="Click or drag to seek"
        onPointerDown=${(e) => { e.currentTarget.setPointerCapture(e.pointerId); seekFromPointer(e); }}
        onPointerMove=${(e) => { if (e.buttons) seekFromPointer(e); }}></canvas>

      <div className="pv-bar">
        <button type="button" title="Restart (Home)" onClick=${() => seek(0)}>|◀</button>
        <button type="button" title="Step back (Left arrow; hold Shift for 10x)" onClick=${() => stepFrames(-1)}>◀</button>
        <button type="button" className="pv-play" title="Play / pause (Space)" onClick=${togglePlay}>${playing ? '❚❚' : '▶'}</button>
        <button type="button" title="Step forward (Right arrow; hold Shift for 10x)" onClick=${() => stepFrames(1)}>▶</button>
        <button type="button" title="Jump to end (End)" onClick=${jumpToEnd}>▶|</button>
        <input type="range" className="pv-scrub" min="0" max="1000" aria-label="Timeline"
          value=${total ? Math.round((1000 * hud.time) / total) : 0}
          onChange=${(e) => seek((total * e.target.value) / 1000)} />
        <span className="pv-time">${hud.time.toFixed(1)} / ${total.toFixed(1)} s</span>
      </div>

      <div className="pv-bar pv-opts">
        <label>Speed
          <select value=${speed} onChange=${(e) => setSpeed(parseFloat(e.target.value))}>
            ${SPEEDS.map((s) => html`<option key=${s} value=${s}>${s}x</option>`)}
          </select>
        </label>
        <label><input type="checkbox" checked=${loop} onChange=${(e) => setLoop(e.target.checked)} /> loop</label>
        <span className="pv-sep"></span>
        <${Segmented} className="pv-seg" label="Camera" value=${cameraMode} onChange=${chooseCamera}
          options=${[{ value: 'orbit', label: 'Orbit' }, { value: 'follow', label: 'Follow' }, { value: 'chase', label: 'Chase' }]} />
        <button type="button" className="pv-mini" onClick=${() => showView('fit')}>Fit</button>
        <button type="button" className="pv-mini" onClick=${() => showView('top')}>Top</button>
        <button type="button" className="pv-mini" onClick=${() => showView('side')}>Side</button>
        <span className="pv-sep"></span>
        ${OPTION_LABELS.filter(([name]) => !(needsChasers.includes(name) && !(track && track.threats)) && !(needsForces.includes(name) && !(track && track.dyn))).map(([name, label]) => html`
          <label key=${name}><input type="checkbox" checked=${options[name]} onChange=${() => toggleOption(name)} /> ${label}</label>`)}
      </div>
    </div>`;
}
