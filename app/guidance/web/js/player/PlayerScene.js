// The three.js part of the 3D player: scene, drone model, path lines, camera, and the little distance/speed chart.
// It knows nothing about React; components/Player.js owns the buttons and calls the methods below.
//
//   const scene = new PlayerScene(canvas, chartCanvas);
//   scene.load(track, meta);                       // a new run
//   scene.draw(time, secondsSinceLastFrame, options, cameraMode);   // call once per animation frame
//
// `track` is what the server sends: { n, dt, stride, steps, pos, quat, phase, dist, speed } with one entry per frame
// (phase 0 = scripted controller flying, 1 = trained policy flying). The simulator is z-up.
// With forces recorded the track also has `dyn` (per frame: rotor thrusts actual / wanted, total thrust, torques, drag, net force,
// accel, action and the angles to the target) and `dyn_meta` (constants), drawn as vectors and shown in the side panel.
// With chasing drones the track also has `threats` (per frame, one [x, y, z] per chaser) and `threat_dist`
// (distance from the drone to the nearest chaser).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);          // the simulator is z-up; this lets its quaternions apply untouched

export const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64];
const COLOR_CONTROLLER = 0x4c8dff;
const COLOR_POLICY = 0xff9f43;
const COLOR_TARGET = 0xff4d5e;
const COLOR_CHASER = 0xe5383b;
// one colour per chaser slot, the same order as THREAT_COLORS in base_drone_env_isaac.py (the learner itself is red there)
export const CHASER_COLORS = [0x1a59f2, 0x1acc40, 0xf2bf1a, 0xb340e6, 0x1ad9d9, 0xff8000];
// The sim's rotor order (mixer / rotor_f arrays): body positions at 45, 135, 225, 315 deg, spins +1, -1, +1, -1.
export const ROTOR_SIGNS = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
export const ROTOR_COLORS = [0xff6b6b, 0x4cc27f, 0x5aa9ff, 0xf7c948];
const COLOR_THRUST = 0xff9f43, COLOR_THRUST_WANT = 0xfff0a0, COLOR_WEIGHT = 0x9aa5bd, COLOR_DRAG = 0xb07cff, COLOR_NET = 0x4cffa0;
const COLOR_ARC_THRUST = 0xc6f24e, COLOR_ARC_VEL = 0x6ad4ff;
const ROTOR_ARROW_PER_HOVER = 0.45;        // arrow length (in drone-model units) of one rotor's hover thrust
const FORCE_ARROW_PER_WEIGHT = 1.6;        // arrow length (in drone-model units) of a force equal to the drone's weight
const ARC_POINTS = 25;
const UP = new THREE.Vector3(0, 0, 1), DOWN = new THREE.Vector3(0, 0, -1);

const COLOR_CHASER_CSS = '#ff6b6b';
const clamp = (x, low, high) => Math.min(Math.max(x, low), high);
const cssColor = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';

// 1, 2, 5, 10, 20, 50 ... the next "round" number at or below x (used for grid sizes)
function niceStep(x) {
  const power = 10 ** Math.floor(Math.log10(x));
  const mantissa = x / power;
  return (mantissa < 1.5 ? 1 : mantissa < 3.5 ? 2 : mantissa < 7.5 ? 5 : 10) * power;
}

export class PlayerScene {
  constructor(canvas, chartCanvas) {
    this.chart = chartCanvas;
    this.run = null;                 // the loaded run (null until load())
    this.extent = 10;                // rough size of the flight in metres, used to scale the camera and markers
    this.chaseDirection = new THREE.Vector3(1, 0, 0);
    this.sampled = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), velocity: new THREE.Vector3() };

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d111b);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 1e6);
    this.camera.position.set(-20, -30, 20);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.zoomToCursor = true;
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = 1e5;
    this.controls.screenSpacePanning = true;

    this.scene.add(new THREE.HemisphereLight(0xcfe0ff, 0x1a2030, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(30, -20, 60);
    this.scene.add(sun);

    this.world = new THREE.Group();            // everything that is rebuilt for each run (grid, paths, target ...)
    this.scene.add(this.world);
    this.buildDrone();
    this.buildHelpers();
  }

  // ---------------------------------------------------------------- one-time objects
  buildDrone() {
    this.drone = new THREE.Group();
    this.rotors = [];
    this.bodyMaterial = new THREE.MeshStandardMaterial({ color: COLOR_CONTROLLER, metalness: 0.2, roughness: 0.5 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x20252f, roughness: 0.7 });
    this.drone.add(new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.2, 0.1), this.bodyMaterial));
    for (const sign of [1, -1]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.04, 0.03), dark);
      arm.rotation.z = sign * Math.PI / 4;
      this.drone.add(arm);
    }
    for (const [x, y] of ROTOR_SIGNS) {
      const rotor = new THREE.Mesh(
        new THREE.CylinderGeometry(0.17, 0.17, 0.012, 24),
        new THREE.MeshStandardMaterial({ color: 0x9fb4d8, transparent: true, opacity: 0.55 }),
      );
      rotor.rotation.x = Math.PI / 2;
      rotor.position.set(x * 0.265, y * 0.265, 0.045);
      this.drone.add(rotor);
      this.rotors.push(rotor);
    }
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.18, 12), new THREE.MeshStandardMaterial({ color: 0xffffff }));
    nose.rotation.z = -Math.PI / 2;
    nose.position.set(0.22, 0, 0);            // +x is "forward"
    this.drone.add(nose);
    // one coloured arrow (the thrust the rotor makes) and one pale arrow (what the mixer asks of it; red and pointing down
    // when it asks for a push the rotor cannot make) per rotor, in the drone's own frame so they tilt with it
    this.rotorArrows = ROTOR_SIGNS.map(([x, y], i) => {
      const origin = new THREE.Vector3(x * 0.265, y * 0.265, 0.06);
      const actual = new THREE.ArrowHelper(UP, origin, 0.4, ROTOR_COLORS[i], 0.1, 0.07);
      const wanted = new THREE.ArrowHelper(UP, origin, 0.4, 0xffffff, 0.08, 0.04);
      for (const part of [wanted.line.material, wanted.cone.material]) { part.transparent = true; part.opacity = 0.6; }
      actual.visible = wanted.visible = false;
      this.drone.add(actual, wanted);
      return { actual, wanted };
    });
    this.drone.visible = false;
    this.scene.add(this.drone);
  }

  buildHelpers() {
    const twoPoints = () => new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    this.velocityArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0xffffff, 0.3, 0.15);
    this.guideLine = new THREE.Line(twoPoints(), new THREE.LineDashedMaterial({ color: COLOR_TARGET, dashSize: 1, gapSize: 1, transparent: true, opacity: 0.6 }));
    this.dropLine = new THREE.Line(twoPoints(), new THREE.LineBasicMaterial({ color: 0x8fa0c0, transparent: true, opacity: 0.5 }));
    this.shadow = new THREE.Mesh(new THREE.CircleGeometry(0.35, 20), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.55 }));
    this.scene.add(this.velocityArrow, this.guideLine, this.dropLine, this.shadow);

    // whole-drone force vectors (world frame, drawn at the drone) and the two angle arcs towards the target line
    const makeArrow = (color) => {
      const arrow = new THREE.ArrowHelper(UP, new THREE.Vector3(), 1, color, 0.3, 0.15);
      arrow.visible = false;
      this.scene.add(arrow);
      return arrow;
    };
    this.forceArrows = {
      thrust: makeArrow(COLOR_THRUST), thrustWant: makeArrow(COLOR_THRUST_WANT), weight: makeArrow(COLOR_WEIGHT),
      drag: makeArrow(COLOR_DRAG), net: makeArrow(COLOR_NET),
    };
    const makeArc = (color) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ARC_POINTS * 3), 3));
      const arc = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color }));
      arc.visible = false;
      arc.frustumCulled = false;
      this.scene.add(arc);
      return arc;
    };
    this.arcs = { thrust: makeArc(COLOR_ARC_THRUST), vel: makeArc(COLOR_ARC_VEL) };
  }

  // ---------------------------------------------------------------- loading a run
  get duration() {
    return this.run ? (this.run.n - 1) * this.run.dt : 0;
  }

  // Builds everything for a new run and returns the playback speed that makes the replay last about 30 s.
  load(track, meta) {
    this.clearWorld();
    const n = track.pos.length;
    const run = {
      n, dt: track.dt, meta,
      pos: track.pos.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
      quat: track.quat.map((q) => new THREE.Quaternion(q[0], q[1], q[2], q[3])),
      phase: track.phase, dist: track.dist, speed: track.speed, steps: track.steps,
      target: new THREE.Vector3(...meta.target_pos),
      // moving target (acro --runner): the target's position per frame; run.target is then moved to the current one every frame
      targets: track.target && track.target.length ? track.target.map((p) => new THREE.Vector3(p[0], p[1], p[2])) : null,
      start: new THREE.Vector3(...meta.start_pos),
      chasers: track.threats && track.threats.length ? track.threats.map((frame) => frame.map((p) => new THREE.Vector3(p[0], p[1], p[2]))) : null,
      chaserDist: track.threat_dist || null,
      // per frame, one entry per chaser: speed m/s, velocity, and deg between its velocity and its line to the drone (null while it hovers)
      chaserSpeed: track.threat_speed || null, chaserVel: track.threat_vel || null, chaserLos: track.threat_los_deg || null,
      chaserRadius: track.threat_radius || 0.3,
      dyn: track.dyn || null, dynMeta: track.dyn_meta || null,
      // line of sight: angle between the velocity and the direction to the target, the rc reward's los term and its running total
      los: track.los_deg ? { deg: track.los_deg, reward: track.los_reward, sum: track.los_reward_sum, k: track.los_meta ? track.los_meta.k : null } : null,
    };
    // velocity of each frame from its neighbours (central difference), for the velocity arrow
    run.vel = run.pos.map((_, i) => {
      const before = Math.max(i - 1, 0), after = Math.min(i + 1, n - 1);
      return run.pos[after].clone().sub(run.pos[before]).divideScalar(Math.max(track.dt * (after - before), 1e-6));
    });
    run.handoffFrame = track.phase.indexOf(1);          // first frame flown by the policy (-1 = never)
    this.run = run;

    this.buildGround();
    this.buildPaths();
    this.buildTargetMarkers();
    this.buildRunnerTrail();
    this.buildChasers();

    this.controls.maxDistance = Math.max(this.extent * 40, 100);
    this.camera.far = Math.max(this.extent * 200, 1e3);
    this.camera.updateProjectionMatrix();
    this.drone.visible = true;
    this.showView('fit');
    this.drawChart(0);
    return SPEEDS.find((s) => this.duration / s <= 30) || SPEEDS[SPEEDS.length - 1];
  }

  clearWorld() {
    for (const child of [...this.world.children]) {
      this.world.remove(child);
      child.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material && object.material.dispose) object.material.dispose();
      });
    }
  }

  buildGround() {
    const { run } = this;
    const chaserPoints = run.chasers ? run.chasers.flat() : [];
    const box = new THREE.Box3().setFromPoints([...run.pos, ...chaserPoints, ...(run.targets || []), run.target, new THREE.Vector3(run.target.x, run.target.y, 0)]);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    this.extent = Math.max(size.length() * 0.5, 3);

    const gridSize = niceStep(this.extent * 4);
    const gridDivisions = Math.max(10, Math.min(80, Math.round(gridSize / niceStep(this.extent / 6))));
    const grid = new THREE.GridHelper(gridSize, gridDivisions, 0x3b4766, 0x232b3f);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(center.x, center.y, 0);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(gridSize, gridSize), new THREE.MeshBasicMaterial({ color: 0x121829, transparent: true, opacity: 0.8 }));
    floor.position.set(center.x, center.y, -0.02);
    this.world.add(grid, floor, new THREE.AxesHelper(Math.max(2, this.extent * 0.1)));
  }

  // Two copies of the flight path coloured by who was flying: a faint full path and the trail that grows while playing.
  buildPaths() {
    const { run } = this;
    const makeGeometry = () => {
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(run.n * 3);
      const colors = new Float32Array(run.n * 3);
      const controllerColor = new THREE.Color(COLOR_CONTROLLER);
      const policyColor = new THREE.Color(COLOR_POLICY);
      for (let i = 0; i < run.n; i++) {
        positions.set([run.pos[i].x, run.pos[i].y, run.pos[i].z], i * 3);
        (run.phase[i] ? policyColor : controllerColor).toArray(colors, i * 3);
      }
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      return geometry;
    };
    this.ghostLine = new THREE.Line(makeGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.22 }));
    this.trailLine = new THREE.Line(makeGeometry(), new THREE.LineBasicMaterial({ vertexColors: true }));
    this.world.add(this.ghostLine, this.trailLine);
  }

  // The target: a true-size hit sphere, a screen-size beacon (so it can be found from far away), the switch-distance
  // sphere (where the policy takes over) and a marker at the start.
  buildTargetMarkers() {
    const { run } = this;
    const target = run.target;
    this.hitSphere = new THREE.Mesh(new THREE.SphereGeometry(run.meta.hit_radius || 0.25, 24, 16), new THREE.MeshBasicMaterial({ color: COLOR_TARGET, transparent: true, opacity: 0.55 }));
    this.beacon = new THREE.Mesh(new THREE.OctahedronGeometry(1), new THREE.MeshBasicMaterial({ color: COLOR_TARGET, wireframe: true }));
    this.switchSphere = new THREE.Mesh(new THREE.SphereGeometry(run.meta.switch_dist || 0.01, 24, 12), new THREE.MeshBasicMaterial({ color: 0x4cc27f, wireframe: true, transparent: true, opacity: 0.07 }));
    this.startMarker = new THREE.Mesh(new THREE.OctahedronGeometry(1), new THREE.MeshBasicMaterial({ color: 0x8fa0c0, wireframe: true }));
    for (const marker of [this.hitSphere, this.beacon, this.switchSphere]) marker.position.copy(target);
    this.startMarker.position.copy(run.start);
    this.world.add(this.hitSphere, this.beacon, this.switchSphere, this.startMarker);
  }

  // Moving target: the path it flees along (the hit sphere and beacon follow it in drawTarget).
  buildRunnerTrail() {
    const { run } = this;
    this.runnerTrail = null;
    if (!run.targets) return;
    // escape sphere: the runner is lost once it leaves this sphere around its spawn point
    const runner = run.meta.runner;
    if (runner && runner.spawn && runner.escape_dist) {
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(runner.escape_dist, 32, 16),
        new THREE.MeshBasicMaterial({ color: COLOR_TARGET, wireframe: true, transparent: true, opacity: 0.08 }));
      sphere.position.set(...runner.spawn);
      this.world.add(sphere);
    }
    const positions = new Float32Array(run.n * 3);
    run.targets.forEach((p, i) => positions.set([p.x, p.y, p.z], i * 3));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.runnerTrail = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: COLOR_TARGET, transparent: true, opacity: 0.6 }));
    this.world.add(this.runnerTrail);
  }

  // Moves the target (and everything that reads run.target: guide line, angle arcs) to where it is at this moment.
  drawTarget(info) {
    const { run } = this;
    if (!run.targets) return;
    run.target.lerpVectors(run.targets[info.index], run.targets[info.next], info.blend);
    this.hitSphere.position.copy(run.target);
    this.beacon.position.copy(run.target);
    this.runnerTrail.geometry.setDrawRange(0, info.index + 2);
  }

  // One red marker plus a trail per chasing drone (none for a run without chasers).
  buildChasers() {
    const { run } = this;
    this.chasers = [];
    if (!run.chasers) return;
    const count = run.chasers[0].length;
    for (let k = 0; k < count; k++) {
      const color = CHASER_COLORS[k % CHASER_COLORS.length];
      const body = new THREE.Mesh(new THREE.SphereGeometry(run.chaserRadius, 16, 12), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.25, roughness: 0.5 }));
      const positions = new Float32Array(run.n * 3);
      run.chasers.forEach((frame, i) => positions.set([frame[k].x, frame[k].y, frame[k].z], i * 3));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const trail = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }));
      // its LOS arc: from its velocity to its line to the drone
      const arcGeometry = new THREE.BufferGeometry();
      arcGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ARC_POINTS * 3), 3));
      const arc = new THREE.Line(arcGeometry, new THREE.LineBasicMaterial({ color }));
      arc.visible = false;
      arc.frustumCulled = false;
      this.world.add(body, trail, arc);
      this.chasers.push({ body, trail, arc });
    }
  }

  // ---------------------------------------------------------------- per-frame work
  // The drone's state at `time` seconds, blended between the two nearest frames. Also fills this.sampled.
  sample(time) {
    const { run, sampled } = this;
    const frame = clamp(time / run.dt, 0, run.n - 1);
    const i = Math.floor(frame), j = Math.min(i + 1, run.n - 1), blend = frame - i;
    sampled.position.lerpVectors(run.pos[i], run.pos[j], blend);
    sampled.quaternion.slerpQuaternions(run.quat[i], run.quat[j], blend);
    sampled.velocity.lerpVectors(run.vel[i], run.vel[j], blend);
    return {
      index: i,
      dist: run.dist[i] + (run.dist[j] - run.dist[i]) * blend,
      speed: run.speed[i] + (run.speed[j] - run.speed[i]) * blend,
      chaserDist: run.chaserDist ? run.chaserDist[i] + (run.chaserDist[j] - run.chaserDist[i]) * blend : null,
      // per chaser: distance to the drone (from the blended positions), speed, LOS deg
      chasers: run.chasers ? run.chasers[i].map((p, k) => ({
        dist: p.clone().lerp(run.chasers[j][k], blend).distanceTo(sampled.position),
        speed: run.chaserSpeed ? run.chaserSpeed[i][k] : null,
        los: run.chaserLos ? run.chaserLos[i][k] : null,
      })) : null,
      los: run.los ? { deg: run.los.deg[i], reward: run.los.reward[i], sum: run.los.sum[i], k: run.los.k } : null,
      blend, next: j,
    };
  }

  // Draws one frame. `options` = { trail, ghost, vel, guide, drops, switch, chasers } (booleans), `cameraMode` = orbit | follow | chase.
  // Returns what the HUD shows: { step, dist, speed, policyFlying, chaserDist, los }.
  draw(time, secondsSinceLastFrame, options, cameraMode) {
    const { run, sampled } = this;
    if (!run) return null;
    const info = this.sample(time);
    const position = sampled.position;

    this.drone.position.copy(position);
    this.drone.quaternion.copy(sampled.quaternion);
    const cameraDistance = this.camera.position.distanceTo(position);
    const droneScale = Math.max(1, cameraDistance * 0.04);          // keep the drone visible from far away
    this.drone.scale.setScalar(droneScale);
    this.bodyMaterial.color.setHex(run.phase[info.index] ? COLOR_POLICY : COLOR_CONTROLLER);
    this.rotors.forEach((rotor, n) => { rotor.rotation.y += (n % 2 ? -1 : 1) * 40 * secondsSinceLastFrame; });

    const markerScale = Math.max(0.4, cameraDistance * 0.012);
    this.beacon.scale.setScalar(markerScale);
    this.startMarker.scale.setScalar(markerScale * 0.6);
    this.trailLine.visible = options.trail;
    this.trailLine.geometry.setDrawRange(0, info.index + 2);
    this.ghostLine.visible = options.ghost;
    this.switchSphere.visible = options.switch;

    this.drawTarget(info);
    this.drawChasers(info, options.chasers, cameraDistance, options.arcs);
    const dynamics = this.drawDynamics(info.index, options, droneScale);
    this.drawVelocityArrow(options.vel, droneScale);
    this.drawGuideLine(options.guide, cameraDistance);
    this.drawGroundDrop(options.drops, droneScale);
    this.moveCamera(cameraMode, secondsSinceLastFrame, cameraDistance);

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.drawChart(time);
    return { step: run.steps[info.index], dist: info.dist, speed: info.speed, policyFlying: !!run.phase[info.index], chaserDist: info.chaserDist, chasers: info.chasers, los: info.los, dyn: dynamics };
  }

  // Rotor, force and angle vectors for frame `i`; returns that frame's numbers for the side panel (null without recorded forces).
  drawDynamics(i, options, droneScale) {
    const { run } = this;
    const dyn = run.dyn;
    const showRotor = !!dyn && !!options.rotorVec, showForce = !!dyn && !!options.forceVec, showArcs = !!dyn && !!options.arcs;
    for (const rotor of this.rotorArrows) rotor.actual.visible = rotor.wanted.visible = showRotor;
    for (const arrow of Object.values(this.forceArrows)) arrow.visible = false;
    this.arcs.thrust.visible = this.arcs.vel.visible = false;
    if (!dyn) return null;

    const meta = run.dynMeta;
    const position = this.sampled.position;
    const thrustAxis = UP.clone().applyQuaternion(this.drone.quaternion);

    if (showRotor) {
      this.rotorArrows.forEach((rotor, k) => {
        const wanted = dyn.rotor_f_want[i][k];
        this.setLocalArrow(rotor.actual, dyn.rotor_f[i][k] / meta.rotor_hover_f * ROTOR_ARROW_PER_HOVER);
        this.setLocalArrow(rotor.wanted, wanted / meta.rotor_hover_f * ROTOR_ARROW_PER_HOVER);
        rotor.wanted.setColor(wanted < 0 ? 0xff3b3b : 0xffffff);
      });
    }
    if (showForce) {
      const place = (arrow, vector) => this.placeArrow(arrow, position, vector, meta.weight, droneScale);
      place(this.forceArrows.thrust, thrustAxis.clone().multiplyScalar(dyn.thrust[i]));
      place(this.forceArrows.thrustWant, thrustAxis.clone().multiplyScalar(dyn.thrust_want[i]));
      place(this.forceArrows.weight, new THREE.Vector3(0, 0, -meta.weight));
      place(this.forceArrows.drag, new THREE.Vector3(...dyn.drag[i]));
      place(this.forceArrows.net, new THREE.Vector3(...dyn.f_net[i]));
    }
    if (showArcs) {
      const toTarget = run.target.clone().sub(position).normalize();
      this.setArc(this.arcs.thrust, position, thrustAxis, toTarget, 1.4 * droneScale);
      const velocity = this.sampled.velocity;
      if (velocity.length() > 0.3) this.setArc(this.arcs.vel, position, velocity.clone().normalize(), toTarget, 1.9 * droneScale);
    }

    return {
      rotorF: dyn.rotor_f[i], rotorWant: dyn.rotor_f_want[i], rotorW: dyn.rotor_w[i], thrust: dyn.thrust[i], thrustWant: dyn.thrust_want[i],
      torque: dyn.torque[i], torqueWant: dyn.torque_want[i], drag: dyn.drag[i], net: dyn.f_net[i], accel: dyn.accel[i], action: dyn.action[i],
      angles: Object.fromEntries(Object.entries(dyn.angles).map(([name, values]) => [name, values[i]])),
    };
  }

  // A rotor arrow in the drone's frame: signed length along its +z (negative = pointing down).
  setLocalArrow(arrow, signedLength) {
    const length = Math.abs(signedLength);
    arrow.visible = length > 0.01;
    if (!arrow.visible) return;
    arrow.setDirection(signedLength < 0 ? DOWN : UP);
    arrow.setLength(length, Math.min(0.1, length * 0.5), 0.06);
  }

  // A world-frame force arrow at `origin`; a force equal to the weight is FORCE_ARROW_PER_WEIGHT model units long.
  placeArrow(arrow, origin, vector, weight, droneScale) {
    const length = (vector.length() / weight) * FORCE_ARROW_PER_WEIGHT * droneScale;
    arrow.visible = length > 0.02 * droneScale;
    if (!arrow.visible) return;
    arrow.position.copy(origin);
    arrow.setDirection(vector.clone().normalize());
    arrow.setLength(length, Math.min(0.3 * droneScale, length * 0.4), 0.14 * droneScale);
  }

  // Fills `arc` with the circular arc from direction `from` to direction `to`, around `origin`.
  setArc(arc, origin, from, to, radius) {
    const points = arc.geometry.attributes.position;
    const direction = new THREE.Vector3();
    for (let k = 0; k < ARC_POINTS; k++) {
      direction.lerpVectors(from, to, k / (ARC_POINTS - 1));
      if (direction.lengthSq() < 1e-8) direction.copy(from);        // exactly opposite: no defined arc
      direction.normalize().multiplyScalar(radius).add(origin);
      points.setXYZ(k, direction.x, direction.y, direction.z);
    }
    points.needsUpdate = true;
    arc.visible = true;
  }

  // The chasing drones at the current moment (blended between frames), kept visible from far away like the drone.
  drawChasers(info, visible, cameraDistance, showArc) {
    const { run } = this;
    for (const [k, chaser] of this.chasers.entries()) {
      chaser.body.visible = chaser.trail.visible = !!visible;
      if (!visible) continue;
      chaser.body.position.lerpVectors(run.chasers[info.index][k], run.chasers[info.next][k], info.blend);
      chaser.body.scale.setScalar(Math.max(1, cameraDistance * 0.03));
      chaser.trail.geometry.setDrawRange(0, info.index + 2);
    }
    // each chaser's LOS arc, at the chaser: from its velocity to its line to the drone (like the drone's blue arc)
    for (const [k, chaser] of this.chasers.entries()) {
      chaser.arc.visible = false;
      if (!visible || !showArc || !run.chaserVel || run.chaserLos[info.index][k] == null) continue;
      const toDrone = this.sampled.position.clone().sub(chaser.body.position);
      if (toDrone.lengthSq() < 1e-8) continue;
      const velocity = new THREE.Vector3(...run.chaserVel[info.index][k]).normalize();
      this.setArc(chaser.arc, chaser.body.position, velocity, toDrone.normalize(), Math.max(1.9, cameraDistance * 0.05));
    }
  }

  drawVelocityArrow(visible, droneScale) {
    const velocity = this.sampled.velocity;
    const speed = velocity.length();
    this.velocityArrow.visible = visible && speed > 0.2;
    if (!this.velocityArrow.visible) return;
    this.velocityArrow.position.copy(this.sampled.position);
    this.velocityArrow.setDirection(velocity.clone().normalize());
    this.velocityArrow.setLength(clamp(speed * 0.15, 1, 8) * droneScale, 0.35 * droneScale, 0.18 * droneScale);
  }

  drawGuideLine(visible, cameraDistance) {
    this.guideLine.visible = visible;
    if (!visible) return;
    const { position } = this.sampled;
    const target = this.run.target;
    const points = this.guideLine.geometry.attributes.position;
    points.setXYZ(0, position.x, position.y, position.z);
    points.setXYZ(1, target.x, target.y, target.z);
    points.needsUpdate = true;
    this.guideLine.computeLineDistances();
    this.guideLine.material.dashSize = this.guideLine.material.gapSize = Math.max(cameraDistance * 0.02, 0.2);
    this.guideLine.geometry.computeBoundingSphere();
  }

  // A line straight down to the ground and a shadow dot, to read the altitude.
  drawGroundDrop(visible, droneScale) {
    this.dropLine.visible = this.shadow.visible = visible;
    if (!visible) return;
    const { position } = this.sampled;
    const points = this.dropLine.geometry.attributes.position;
    points.setXYZ(0, position.x, position.y, position.z);
    points.setXYZ(1, position.x, position.y, 0);
    points.needsUpdate = true;
    this.dropLine.geometry.computeBoundingSphere();
    this.shadow.position.set(position.x, position.y, 0.01);
    this.shadow.scale.setScalar(droneScale);
  }

  moveCamera(cameraMode, secondsSinceLastFrame, cameraDistance) {
    if (cameraMode === 'orbit') return;
    const { position, velocity } = this.sampled;
    if (cameraMode === 'follow') {                     // keep the same view angle, just move along with the drone
      const delta = position.clone().sub(this.controls.target);
      this.camera.position.add(delta);
    } else {                                           // chase: sit behind the drone, looking where it flies
      if (velocity.length() > 0.5) {
        this.chaseDirection.lerp(velocity.clone().normalize(), clamp(secondsSinceLastFrame * 4, 0, 1)).normalize();
      }
      const behind = clamp(cameraDistance, 3, this.extent * 2);
      const wanted = position.clone().addScaledVector(this.chaseDirection, -behind).add(new THREE.Vector3(0, 0, behind * 0.3));
      this.camera.position.lerp(wanted, clamp(secondsSinceLastFrame * 5, 0, 1));
    }
    this.controls.target.copy(position);
  }

  // Starts a chase camera looking along the current flight direction.
  startChase(time) {
    if (!this.run) return;
    this.sample(time);
    this.chaseDirection.copy(this.sampled.velocity).setZ(0);
    if (this.chaseDirection.lengthSq() < 1e-6) this.chaseDirection.set(1, 0, 0);
    this.chaseDirection.normalize();
  }

  // Puts the camera at a preset angle that shows the whole flight: 'fit' | 'top' | 'side'.
  showView(kind) {
    if (!this.run) return;
    const center = new THREE.Box3().setFromPoints([...this.run.pos, ...(this.run.targets || []), this.run.target]).getCenter(new THREE.Vector3());
    const distance = this.extent / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.05;
    const directions = {
      fit: new THREE.Vector3(-0.6, -0.8, 0.5),
      top: new THREE.Vector3(0.0001, -0.0001, 1),
      side: new THREE.Vector3(0, -1, 0.05),
    };
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(directions[kind].normalize(), distance);
    this.controls.update();
  }

  resize(width, height) {
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  // ---------------------------------------------------------------- distance / speed chart under the 3D view
  drawChart(time) {
    const { run, chart } = this;
    if (!run) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = chart.clientWidth, height = chart.clientHeight;
    if (!width || !height) return;
    if (chart.width !== Math.round(width * ratio)) {
      chart.width = Math.round(width * ratio);
      chart.height = Math.round(height * ratio);
    }
    const g = chart.getContext('2d');
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    g.clearRect(0, 0, width, height);
    g.fillStyle = cssColor('--soft');
    g.fillRect(0, 0, width, height);

    const x = (i) => (i / Math.max(run.n - 1, 1)) * (width - 2) + 1;
    const maxDist = Math.max(...run.dist, ...(run.chaserDist || []), 1);
    const maxSpeed = Math.max(...run.speed, 1);
    const logDist = (d) => Math.log1p(d) / Math.log1p(maxDist);
    const drawLine = (values, normalise, color) => {
      g.beginPath();
      values.forEach((value, i) => {
        const y = height - 4 - normalise(value) * (height - 8);
        if (i) g.lineTo(x(i), y); else g.moveTo(x(i), y);
      });
      g.strokeStyle = color;
      g.lineWidth = 1.5;
      g.stroke();
    };

    if (run.handoffFrame >= 0) {                       // shade the part flown by the policy
      g.fillStyle = 'rgba(255,159,67,0.12)';
      g.fillRect(x(run.handoffFrame), 0, width - x(run.handoffFrame), height);
    }
    drawLine(run.speed, (v) => v / maxSpeed, 'rgba(255,159,67,0.9)');
    drawLine(run.dist, logDist, '#6c8cff');
    if (run.chaserDist) drawLine(run.chaserDist, logDist, COLOR_CHASER_CSS);
    g.fillStyle = cssColor('--muted');
    g.font = '10.5px sans-serif';
    g.fillText(`distance (log, max ${maxDist.toFixed(0)} m)`, 6, 12);
    g.fillText(`speed (max ${maxSpeed.toFixed(1)} m/s)`, 6, 24);
    if (run.chaserDist) { g.fillStyle = COLOR_CHASER_CSS; g.fillText('nearest chaser (same log scale)', 6, 36); g.fillStyle = cssColor('--muted'); }
    if (run.handoffFrame >= 0) g.fillText('RL takes over', x(run.handoffFrame) + 4, height - 6);

    const cursor = (time / Math.max(this.duration, 1e-9)) * (width - 2) + 1;      // the "now" line
    g.strokeStyle = cssColor('--ink');
    g.lineWidth = 1.5;
    g.beginPath();
    g.moveTo(cursor, 0);
    g.lineTo(cursor, height);
    g.stroke();
  }

  dispose() {
    this.clearWorld();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
