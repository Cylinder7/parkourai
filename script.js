/* =========================================================
   NeuroParkour — AI learns a 2D platformer via neuroevolution
   Pure vanilla JS. Single-file logic.
   ========================================================= */

'use strict';

/* ---------- Utility ---------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function sigmoid(x) {
  if (x < -20) return 0;
  if (x > 20) return 1;
  return 1 / (1 + Math.exp(-x));
}
function randn() { // standard normal via Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ---------- Global Settings ---------- */
const settings = {
  // Evolution
  popSize: 50,
  mutRate: 0.10,
  mutAmt: 0.50,
  elitism: 2,
  selPct: 0.30,
  crossover: true,
  // NN
  hidden: [8, 6],
  rayCount: 5,
  rayRange: 180,
  autorun: true,
  // Physics
  gravity: 0.60,
  jumpForce: -12.0,
  runSpeed: 3.5,
  timeout: 20,
  // Level
  seed: 42,
  levelLen: 6000,
  difficulty: 1,
  // Mechanics
  doubleJump: false,
  // Viz
  showRays: true,
  showAll: true,
  showDead: false,
  followCam: true,
};

/* ---------- Neural Network ---------- */
class NeuralNetwork {
  constructor(sizes) {
    // sizes = [in, h1, h2, ..., out]
    this.sizes = sizes.slice();
    this.weights = []; // each: array[out][in]
    this.biases = [];  // each: array[out]
    for (let i = 0; i < sizes.length - 1; i++) {
      const r = sizes[i], c = sizes[i + 1];
      const W = [];
      const B = [];
      for (let o = 0; o < c; o++) {
        const row = new Float32Array(r);
        for (let k = 0; k < r; k++) row[k] = randn() * 1.5;
        W.push(row);
        B.push(randn() * 1.0);
      }
      this.weights.push(W);
      this.biases.push(B);
    }
  }

  forward(input) {
    let cur = input;
    for (let i = 0; i < this.weights.length; i++) {
      const W = this.weights[i], B = this.biases[i];
      const next = new Float32Array(W.length);
      const isLast = i === this.weights.length - 1;
      for (let o = 0; o < W.length; o++) {
        let s = B[o];
        const row = W[o];
        for (let k = 0; k < row.length; k++) s += cur[k] * row[k];
        next[o] = isLast ? sigmoid(s) : Math.tanh(s);
      }
      cur = next;
    }
    return cur;
  }

  copy() {
    const nn = Object.create(NeuralNetwork.prototype);
    nn.sizes = this.sizes.slice();
    nn.weights = [];
    nn.biases = [];
    for (let i = 0; i < this.weights.length; i++) {
      const W = this.weights[i];
      const cw = [];
      for (let o = 0; o < W.length; o++) {
        cw.push(Float32Array.from(W[o]));
      }
      nn.weights.push(cw);
      nn.biases.push(Float32Array.from(this.biases[i]));
    }
    return nn;
  }

  mutate(rate, amt) {
    for (let i = 0; i < this.weights.length; i++) {
      const W = this.weights[i], B = this.biases[i];
      for (let o = 0; o < W.length; o++) {
        const row = W[o];
        for (let k = 0; k < row.length; k++) {
          if (Math.random() < rate) {
            row[k] += randn() * amt;
            row[k] = clamp(row[k], -10, 10);
          }
        }
        if (Math.random() < rate) {
          B[o] += randn() * amt;
          B[o] = clamp(B[o], -10, 10);
        }
      }
    }
  }

  static crossover(a, b) {
    const nn = a.copy();
    for (let i = 0; i < nn.weights.length; i++) {
      const W = nn.weights[i], B = nn.biases[i];
      const Wb = b.weights[i], Bb = b.biases[i];
      for (let o = 0; o < W.length; o++) {
        if (Math.random() < 0.5) {
          W[o] = Float32Array.from(Wb[o]);
          B[o] = Bb[o];
        }
      }
    }
    return nn;
  }

  serialize() {
    const out = { sizes: this.sizes.slice(), layers: [] };
    for (let i = 0; i < this.weights.length; i++) {
      out.layers.push({
        weights: this.weights[i].map(r => Array.from(r)),
        biases: Array.from(this.biases[i]),
      });
    }
    return out;
  }

  static deserialize(obj) {
    const nn = Object.create(NeuralNetwork.prototype);
    nn.sizes = obj.sizes.slice();
    nn.weights = [];
    nn.biases = [];
    for (const L of obj.layers) {
      nn.weights.push(L.weights.map(r => Float32Array.from(r)));
      nn.biases.push(Float32Array.from(L.biases));
    }
    return nn;
  }
}

/* ---------- Level Generation ---------- */
function generateLevel(seed, length, difficulty) {
  const rng = mulberry32(seed >>> 0);
  const platforms = []; // {x, y, w, h, kind: 'ground'|'wall'|'plat'}
  const spikes = [];    // {x, y, w, h}
  const coins = [];     // {x, y}

  const groundY = 380;
  const baseGroundH = 70;

  // Start platform — long flat run-up (gives agents time to stabilize)
  platforms.push({ x: 0, y: groundY, w: 400, h: baseGroundH, kind: 'ground' });
  let x = 400;

  // Training-wheels first obstacle: always a small, easy gap so gen-1 agents
  // can discover the jump mechanic without hitting a wall immediately.
  {
    const gap = 45 + rng() * 15; // 45-60px (smaller than normal gaps)
    x += gap;
    const pw = 110 + rng() * 40;
    platforms.push({ x, y: groundY, w: pw, h: baseGroundH, kind: 'ground' });
    x += pw;
  }

  // Difficulty scaling
  const gapMin = [40, 55, 70, 90][difficulty];
  const gapMax = [80, 110, 140, 180][difficulty];
  const wallMin = [30, 45, 60, 75][difficulty];
  const wallMax = [60, 90, 120, 150][difficulty];
  const platMin = [80, 70, 60, 50][difficulty];
  const platMax = [140, 120, 100, 90][difficulty];

  let safe = 0; // require some flat ground after hard obstacles
  while (x < length - 200) {
    const r = rng();
    if (r < 0.35 && safe <= 0) {
      // Gap (no platform) — must jump across
      const gap = gapMin + rng() * (gapMax - gapMin);
      x += gap;
      const pw = platMin + rng() * (platMax - platMin);
      const yVar = (rng() - 0.5) * (difficulty >= 2 ? 80 : 40);
      platforms.push({ x, y: clamp(groundY + yVar, 300, 430), w: pw, h: baseGroundH, kind: 'ground' });
      x += pw;
      safe = 1;
    } else if (r < 0.60 && safe <= 0) {
      // Wall — jump over
      const wh = wallMin + rng() * (wallMax - wallMin);
      const ww = 18 + rng() * 10;
      platforms.push({ x: x, y: groundY - wh, w: ww, h: wh + baseGroundH, kind: 'wall' });
      x += ww + 5;
      const pw = platMin + rng() * (platMax - platMin);
      platforms.push({ x, y: groundY, w: pw, h: baseGroundH, kind: 'ground' });
      x += pw;
      safe = 1;
    } else if (r < 0.78 && difficulty >= 1) {
      // Spikes on ground — must jump over
      const sw = 25 + rng() * 35;
      spikes.push({ x: x + 20, y: groundY - 14, w: sw, h: 14 });
      const pw = 60 + rng() * 60;
      platforms.push({ x, y: groundY, w: pw, h: baseGroundH, kind: 'ground' });
      x += pw;
      safe = 0;
    } else if (r < 0.92 && difficulty >= 1) {
      // Floating platform
      const pw = platMin + rng() * (platMax - platMin);
      platforms.push({ x, y: groundY, w: pw, h: baseGroundH, kind: 'ground' });
      x += pw;
      const upperW = 50 + rng() * 50;
      const upperY = groundY - 70 - rng() * 50;
      platforms.push({ x, y: upperY, w: upperW, h: 14, kind: 'plat' });
      // Optional coin
      if (rng() < 0.6) coins.push({ x: x + upperW / 2, y: upperY - 18 });
      x += upperW + 10 + rng() * 30;
      safe = 0;
    } else {
      // Flat ground — breather
      const pw = 120 + rng() * 80;
      platforms.push({ x, y: groundY, w: pw, h: baseGroundH, kind: 'ground' });
      x += pw;
      safe = 0;
    }
  }
  // Final goal platform
  platforms.push({ x, y: groundY, w: 200, h: baseGroundH, kind: 'ground' });
  const goalX = x + 80;

  // Build a spatial index for collision: simple flat list (fast enough)
  return { platforms, spikes, coins, goalX, width: x + 200, groundY };
}

/* ---------- Agent ---------- */
class Agent {
  constructor(brain, level) {
    this.brain = brain;
    this.level = level;
    this.reset();
  }
  reset() {
    this.x = 40;
    this.y = 320;
    this.vx = 0;
    this.vy = 0;
    this.w = 14;
    this.h = 20;
    this.onGround = false;
    this.alive = true;
    this.timeAlive = 0;
    this.maxX = this.x;
    this.jumps = 0;
    this.lastProgressTime = 0;
    this.fitness = 0;
    this.reachedGoal = false;
    this.coyoteTime = 0;
    this.jumpBuffer = 0;
    this.airJumpsLeft = 0;
    // Pre-computed ray angles based on rayCount
    this._rayAngles = this._computeRayAngles(settings.rayCount);
  }
  _computeRayAngles(n) {
    // Bias rays toward forward-down (most useful for detecting gaps & walls ahead).
    // Spread from -30° (up-right) to 90° (straight down).
    // n=5: -30°, 0°, 30°, 60°, 90°  (wall, forward, gap-edge, landing, ground)
    // n=7: -30°, -10°, 10°, 30°, 50°, 70°, 90°
    const angles = [];
    if (n === 1) return [Math.PI / 4];
    const start = -Math.PI / 6;  // -30° (up-right)
    const end = Math.PI / 2;     // 90° (down)
    for (let i = 0; i < n; i++) {
      angles.push(start + (end - start) * (i / (n - 1)));
    }
    return angles;
  }

  /* Cast a ray from (x,y) in direction (dx,dy), return distance to nearest solid (or max) */
  castRay(x, y, dx, dy, maxDist) {
    const step = 6;
    let d = 0;
    while (d < maxDist) {
      const px = x + dx * d;
      const py = y + dy * d;
      if (this.level.isSolidAt(px, py) || this.level.isSpikeAt(px, py)) return d;
      d += step;
    }
    return maxDist;
  }

  sense() {
    const inputs = [];
    const cx = this.x + this.w / 2;
    const cy = this.y + this.h / 2;
    const range = settings.rayRange;
    for (const a of this._rayAngles) {
      const dx = Math.cos(a);
      const dy = Math.sin(a);
      const d = this.castRay(cx, cy, dx, dy, range);
      inputs.push(d / range); // 0 = touching, 1 = nothing
    }
    inputs.push(clamp(this.vy / 12, -1, 1)); // vy normalized
    inputs.push(this.onGround ? 1 : 0);
    return inputs;
  }

  step(dt) {
    if (!this.alive) return;
    this.timeAlive += dt;

    // Brain decision
    const inputs = this.sense();
    const out = this.brain.forward(inputs);

    let moveLeft = false, moveRight = false, jump = false;
    if (settings.autorun) {
      jump = out[0] > 0.5;
      moveRight = true;
    } else {
      jump   = out[0] > 0.5;
      moveRight = out[1] > 0.5;
      moveLeft  = out[2] > 0.5;
    }

    // Horizontal motion
    const target = (moveRight ? 1 : 0) - (moveLeft ? 1 : 0);
    this.vx = lerp(this.vx, target * settings.runSpeed, 0.25);

    // Coyote time: can still jump shortly after leaving a ledge.
    // Jump buffer: if jump is pressed slightly before landing, it fires on landing.
    // Double jump: if enabled, one extra air-jump allowed.
    this.coyoteTime = Math.max(0, this.coyoteTime - dt);
    this.jumpBuffer = Math.max(0, this.jumpBuffer - dt);
    if (this.onGround) {
      this.coyoteTime = 0.12; // 120ms grace after leaving ground
      this.airJumpsLeft = settings.doubleJump ? 1 : 0;
    }
    if (jump) this.jumpBuffer = 0.12;
    if (this.jumpBuffer > 0) {
      if (this.onGround || this.coyoteTime > 0) {
        this.vy = settings.jumpForce;
        this.onGround = false;
        this.coyoteTime = 0;
        this.jumpBuffer = 0;
        this.jumps++;
      } else if (settings.doubleJump && this.airJumpsLeft > 0) {
        this.vy = settings.jumpForce;
        this.airJumpsLeft--;
        this.jumpBuffer = 0;
        this.jumps++;
      }
    }

    // Gravity
    this.vy += settings.gravity * dt * 60; // dt in seconds, scale to per-frame
    if (this.vy > 18) this.vy = 18;

    // Apply horizontal motion with collision
    const dx = this.vx * dt * 60;
    const dy = this.vy * dt * 60;
    this.moveAxis(dx, 0);
    this.moveAxis(0, dy);

    // Check spikes
    if (this.level.rectHitsSpike(this.x, this.y, this.w, this.h)) {
      this.alive = false;
    }
    // Fell off world
    if (this.y > 700) {
      this.alive = false;
    }
    // Goal reached
    if (this.x + this.w > this.level.goalX && !this.reachedGoal) {
      this.reachedGoal = true;
      this.alive = false;
    }

    // Track progress
    if (this.x > this.maxX) {
      this.maxX = this.x;
      this.lastProgressTime = this.timeAlive;
    }

    // Anti-stuck: if no progress for 5s, kill (generous to allow exploration)
    if (this.timeAlive - this.lastProgressTime > 5) {
      this.alive = false;
    }

    // Timeout
    if (this.timeAlive > settings.timeout) {
      this.alive = false;
    }
  }

  moveAxis(dx, dy) {
    // Move along one axis and resolve collisions
    this.x += dx;
    this.y += dy;
    // Resolve collisions with platforms
    for (const p of this.level.platforms) {
      if (this.aabb(p.x, p.y, p.w, p.h)) {
        if (dx > 0) { this.x = p.x - this.w; this.vx = 0; }
        else if (dx < 0) { this.x = p.x + p.w; this.vx = 0; }
        if (dy > 0) { this.y = p.y - this.h; this.vy = 0; this.onGround = true; }
        else if (dy < 0) { this.y = p.y + p.h; this.vy = 0; }
      }
    }
    // After vertical move, if no collision below, not on ground
    if (dy === 0) {
      // Check if there's ground just below us
      let grounded = false;
      const probeY = this.y + this.h + 1;
      for (const p of this.level.platforms) {
        if (this.x + this.w > p.x && this.x < p.x + p.w &&
            probeY >= p.y && probeY <= p.y + p.h) {
          grounded = true; break;
        }
      }
      // Only set onGround=false here if we were moving horizontally;
      // vertical moves already set it correctly.
      if (!grounded) this.onGround = false;
    }
  }

  aabb(px, py, pw, ph) {
    return this.x < px + pw && this.x + this.w > px &&
           this.y < py + ph && this.y + this.h > py;
  }

  computeFitness() {
    // Primary: max horizontal distance reached.
    // Small speed bonus for tie-breaking (faster agents slightly preferred).
    // Huge bonus for reaching the goal.
    let f = this.maxX;
    f -= this.timeAlive * 0.5; // gentle speed incentive (breaks ties toward faster agents)
    if (this.reachedGoal) f += 100000;
    this.fitness = f;
    return f;
  }
}

/* ---------- Level helper methods (added to generated level obj) ---------- */
function attachLevelHelpers(level) {
  level.isSolidAt = function(x, y) {
    for (const p of this.platforms) {
      if (x >= p.x && x < p.x + p.w && y >= p.y && y < p.y + p.h) return true;
    }
    return false;
  };
  level.isSpikeAt = function(x, y) {
    for (const s of this.spikes) {
      if (x >= s.x && x < s.x + s.w && y >= s.y && y < s.y + s.h) return true;
    }
    return false;
  };
  level.rectHitsSpike = function(x, y, w, h) {
    for (const s of this.spikes) {
      if (x < s.x + s.w && x + w > s.x && y < s.y + s.h && y + h > s.y) return true;
    }
    return false;
  };
  return level;
}

/* ---------- Population ---------- */
class Population {
  constructor(level) {
    this.level = level;
    this.generation = 1;
    this.agents = [];
    this.bestEver = null;       // {brain, fitness, generation}
    this.history = [];          // [{gen, best, avg, worst}]
    this.bestFitnessEver = 0;
    this.lastImprovementGen = 1;
    this.stagnation = 0;        // generations since meaningful improvement
    this.stagnationBoost = 1;   // mutation rate multiplier when stagnating
    this._spawn();
  }

  _inputSize() { return settings.rayCount + 2; }
  _outputSize() { return settings.autorun ? 1 : 3; }

  _makeBrain() {
    const sizes = [this._inputSize(), ...settings.hidden, this._outputSize()];
    return new NeuralNetwork(sizes);
  }

  _spawn() {
    this.agents = [];
    for (let i = 0; i < settings.popSize; i++) {
      this.agents.push(new Agent(this._makeBrain(), this.level));
    }
  }

  aliveCount() { return this.agents.filter(a => a.alive).length; }

  step(dt) {
    for (const a of this.agents) if (a.alive) a.step(dt);
  }

  isAllDead() { return this.aliveCount() === 0; }

  leader() {
    let best = null;
    for (const a of this.agents) {
      if (!a.alive) continue;
      if (!best || a.maxX > best.maxX) best = a;
    }
    if (!best) {
      // Use the agent with the highest maxX overall
      for (const a of this.agents) {
        if (!best || a.maxX > best.maxX) best = a;
      }
    }
    return best;
  }

  evolve() {
    // Compute fitness
    const fits = this.agents.map(a => a.computeFitness());
    // Sort descending by fitness
    const sorted = this.agents.slice().sort((a, b) => b.fitness - a.fitness);

    const best = sorted[0];
    const avg = fits.reduce((s, v) => s + v, 0) / fits.length;
    const worst = sorted[sorted.length - 1].fitness;
    this.history.push({ gen: this.generation, best: best.fitness, avg, worst, stagnation: this.stagnation });

    // Update best ever
    if (!this.bestEver || best.fitness > this.bestEver.fitness) {
      this.bestEver = {
        brain: best.brain.copy(),
        fitness: best.fitness,
        generation: this.generation,
        maxX: best.maxX,
        reachedGoal: best.reachedGoal,
      };
    }

    // Stagnation detection: did the best fitness improve meaningfully?
    const improvementThreshold = Math.max(10, this.bestFitnessEver * 0.03); // 3% or at least 10 units
    if (best.fitness > this.bestFitnessEver + improvementThreshold) {
      this.bestFitnessEver = best.fitness;
      this.lastImprovementGen = this.generation;
      this.stagnation = 0;
      this.stagnationBoost = 1;
    } else {
      this.stagnation++;
      // Gradual mutation boost: 2× at 10 gens, 3× at 20, 4× at 30+
      this.stagnationBoost =
        this.stagnation > 30 ? 4 :
        this.stagnation > 20 ? 3 :
        this.stagnation > 10 ? 2 : 1;
    }

    // Build next generation
    const next = [];
    const popSize = settings.popSize;
    const elite = Math.min(settings.elitism, popSize);
    // Elitism: carry over top N brains unchanged
    for (let i = 0; i < elite; i++) {
      next.push(new Agent(sorted[i].brain.copy(), this.level));
    }
    // Selection pool: top selPct% of population
    const poolSize = Math.max(2, Math.floor(popSize * settings.selPct));
    const pool = sorted.slice(0, poolSize);

    while (next.length < popSize) {
      let childBrain;
      if (settings.crossover) {
        const a = this._tournament(pool);
        const b = this._tournament(pool);
        childBrain = NeuralNetwork.crossover(a.brain, b.brain);
      } else {
        const a = this._tournament(pool);
        childBrain = a.brain.copy();
      }
      childBrain.mutate(settings.mutRate * this.stagnationBoost, settings.mutAmt * this.stagnationBoost);
      next.push(new Agent(childBrain, this.level));
    }

    // Random restart: if severely stagnating, replace bottom 30% with fresh random brains
    if (this.stagnation > 35) {
      const replaceCount = Math.floor(popSize * 0.30);
      for (let i = next.length - replaceCount; i < next.length; i++) {
        next[i] = new Agent(this._makeBrain(), this.level);
      }
      this.stagnation = 10; // partial reset — keep some boost but allow recovery
    }

    this.agents = next;
    this.generation++;
  }

  _tournament(pool) {
    // Tournament selection (size 3): pick 3 random from pool, return the fittest.
    // Better than uniform random for maintaining selection pressure + diversity.
    let best = pool[Math.floor(Math.random() * pool.length)];
    for (let i = 1; i < 3; i++) {
      const cand = pool[Math.floor(Math.random() * pool.length)];
      if (cand.fitness > best.fitness) best = cand;
    }
    return best;
  }

  injectBest() {
    if (!this.bestEver) return;
    // Replace worst half with copies of best brain (mutated)
    const sorted = this.agents.slice().sort((a, b) => b.fitness - a.fitness);
    const half = Math.floor(sorted.length / 2);
    for (let i = half; i < sorted.length; i++) {
      const b = this.bestEver.brain.copy();
      b.mutate(settings.mutRate * 2, settings.mutAmt);
      sorted[i].brain = b;
      sorted[i].reset();
    }
    // Re-sort by fitness desc to keep order
    this.agents = sorted;
  }
}

/* ---------- Renderer ---------- */
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const chartCanvas = document.getElementById('chart');
const chartCtx = chartCanvas.getContext('2d');

let camera = { x: 0, y: 0 };

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cRect = chartCanvas.getBoundingClientRect();
  chartCanvas.width = cRect.width * dpr;
  chartCanvas.height = cRect.height * dpr;
  chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resizeCanvas);

function drawLevel(level, cam) {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  // Background parallax: distant stars
  ctx.fillStyle = '#050a14';
  ctx.fillRect(0, 0, W, H);
  // parallax stars
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  const starOffset = (cam.x * 0.2) % 200;
  for (let i = -1; i < W / 200 + 1; i++) {
    for (let j = 0; j < 4; j++) {
      const sx = i * 200 - starOffset + (j * 50);
      const sy = 30 + j * 80;
      ctx.fillRect(sx, sy, 2, 2);
    }
  }
  // Distant hills (parallax)
  ctx.fillStyle = 'rgba(40, 60, 110, 0.5)';
  ctx.beginPath();
  const hillOff = (cam.x * 0.3) % 300;
  ctx.moveTo(0, H);
  for (let i = -1; i < W / 300 + 2; i++) {
    const bx = i * 300 - hillOff;
    ctx.lineTo(bx, H);
    ctx.lineTo(bx + 150, H - 80 - (i % 2) * 20);
    ctx.lineTo(bx + 300, H);
  }
  ctx.lineTo(W, H);
  ctx.closePath();
  ctx.fill();

  // World transform
  ctx.save();
  ctx.translate(-cam.x, -cam.y);

  // Platforms
  for (const p of level.platforms) {
    if (p.x + p.w < cam.x - 50) continue;
    if (p.x > cam.x + W + 50) break;
    if (p.kind === 'ground') {
      const grad = ctx.createLinearGradient(0, p.y, 0, p.y + p.h);
      grad.addColorStop(0, '#3a8c5a');
      grad.addColorStop(0.3, '#2d6f47');
      grad.addColorStop(1, '#1a4530');
      ctx.fillStyle = grad;
      ctx.fillRect(p.x, p.y, p.w, p.h);
      // Top highlight
      ctx.fillStyle = '#5fbf7f';
      ctx.fillRect(p.x, p.y, p.w, 3);
    } else if (p.kind === 'wall') {
      const grad = ctx.createLinearGradient(p.x, 0, p.x + p.w, 0);
      grad.addColorStop(0, '#8a4a3a');
      grad.addColorStop(0.5, '#6d3825');
      grad.addColorStop(1, '#4d2515');
      ctx.fillStyle = grad;
      ctx.fillRect(p.x, p.y, p.w, p.h);
      ctx.fillStyle = 'rgba(255,255,255,0.1)';
      ctx.fillRect(p.x, p.y, 2, p.h);
    } else { // plat
      const grad = ctx.createLinearGradient(0, p.y, 0, p.y + p.h);
      grad.addColorStop(0, '#6a7da8');
      grad.addColorStop(1, '#3a4a72');
      ctx.fillStyle = grad;
      ctx.fillRect(p.x, p.y, p.w, p.h);
      ctx.fillStyle = '#9fb2e0';
      ctx.fillRect(p.x, p.y, p.w, 2);
    }
  }

  // Spikes (triangles)
  for (const s of level.spikes) {
    if (s.x + s.w < cam.x - 50) continue;
    if (s.x > cam.x + W + 50) continue;
    ctx.fillStyle = '#d0d8e8';
    const teeth = Math.max(1, Math.floor(s.w / 8));
    const tw = s.w / teeth;
    for (let i = 0; i < teeth; i++) {
      ctx.beginPath();
      ctx.moveTo(s.x + i * tw, s.y + s.h);
      ctx.lineTo(s.x + i * tw + tw / 2, s.y);
      ctx.lineTo(s.x + (i + 1) * tw, s.y + s.h);
      ctx.closePath();
      ctx.fill();
    }
    ctx.strokeStyle = '#7080a0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < teeth; i++) {
      ctx.moveTo(s.x + i * tw, s.y + s.h);
      ctx.lineTo(s.x + i * tw + tw / 2, s.y);
      ctx.lineTo(s.x + (i + 1) * tw, s.y + s.h);
    }
    ctx.stroke();
  }

  // Coins
  for (const c of level.coins) {
    if (c.x < cam.x - 30 || c.x > cam.x + W + 30) continue;
    ctx.fillStyle = '#ffd24a';
    ctx.beginPath();
    ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#a87800';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Goal flag
  ctx.fillStyle = '#ffb648';
  ctx.fillRect(level.goalX, level.groundY - 80, 4, 80);
  ctx.beginPath();
  ctx.moveTo(level.goalX + 4, level.groundY - 80);
  ctx.lineTo(level.goalX + 30, level.groundY - 70);
  ctx.lineTo(level.goalX + 4, level.groundY - 60);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#001020';
  ctx.font = 'bold 9px monospace';
  ctx.fillText('GOAL', level.goalX + 6, level.groundY - 67);

  ctx.restore();
}

function drawAgents(pop, leader) {
  const cam = camera;
  ctx.save();
  ctx.translate(-cam.x, -cam.y);

  // Dead agents (optional)
  if (settings.showDead) {
    for (const a of pop.agents) {
      if (a.alive) continue;
      ctx.fillStyle = 'rgba(255,84,112,0.25)';
      ctx.fillRect(a.x, a.y, a.w, a.h);
    }
  }

  // Ghosts (alive, non-leader)
  if (settings.showAll) {
    for (const a of pop.agents) {
      if (!a.alive || a === leader) continue;
      ctx.fillStyle = 'rgba(124,255,178,0.18)';
      ctx.fillRect(a.x, a.y, a.w, a.h);
      ctx.strokeStyle = 'rgba(124,255,178,0.4)';
      ctx.lineWidth = 1;
      ctx.strokeRect(a.x, a.y, a.w, a.h);
    }
  }

  // Leader
  if (leader) {
    // Glow
    ctx.shadowColor = '#00d9ff';
    ctx.shadowBlur = 12;
    ctx.fillStyle = '#00d9ff';
    ctx.fillRect(leader.x, leader.y, leader.w, leader.h);
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(leader.x, leader.y, leader.w, leader.h);

    // Eyes
    ctx.fillStyle = '#001020';
    ctx.fillRect(leader.x + 3, leader.y + 4, 2, 3);
    ctx.fillRect(leader.x + 9, leader.y + 4, 2, 3);

    // Rays
    if (settings.showRays) {
      const cx = leader.x + leader.w / 2;
      const cy = leader.y + leader.h / 2;
      ctx.lineWidth = 1;
      for (const a of leader._rayAngles) {
        const dx = Math.cos(a);
        const dy = Math.sin(a);
        const d = leader.castRay(cx, cy, dx, dy, settings.rayRange);
        const hit = d < settings.rayRange;
        const ex = cx + dx * d;
        const ey = cy + dy * d;
        // Line
        ctx.strokeStyle = hit ? 'rgba(255, 61, 139, 0.5)' : 'rgba(0, 217, 255, 0.25)';
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        // Endpoint dot
        ctx.fillStyle = hit ? '#ff3d8b' : '#00d9ff';
        ctx.beginPath();
        ctx.arc(ex, ey, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

function drawChart(history) {
  const W = chartCanvas.clientWidth, H = chartCanvas.clientHeight;
  chartCtx.clearRect(0, 0, W, H);
  chartCtx.fillStyle = 'transparent';
  // Grid
  chartCtx.strokeStyle = 'rgba(255,255,255,0.05)';
  chartCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (H - 20) * (i / 4) + 10;
    chartCtx.beginPath();
    chartCtx.moveTo(40, y);
    chartCtx.lineTo(W - 10, y);
    chartCtx.stroke();
  }
  if (history.length === 0) {
    chartCtx.fillStyle = '#8590b0';
    chartCtx.font = '11px Inter, sans-serif';
    chartCtx.fillText('No data yet — start training to see fitness history.', 50, H / 2);
    return;
  }
  // Plot best, avg, worst
  const maxFit = Math.max(100, ...history.map(h => h.best));
  const xStep = (W - 50) / Math.max(1, history.length - 1);
  const yScale = (H - 20) / maxFit;

  function plotSeries(key, color) {
    chartCtx.strokeStyle = color;
    chartCtx.lineWidth = 1.5;
    chartCtx.beginPath();
    history.forEach((h, i) => {
      const x = 40 + i * xStep;
      const y = H - 10 - h[key] * yScale;
      if (i === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();
  }
  plotSeries('worst', 'rgba(255,84,112,0.5)');
  plotSeries('avg',   'rgba(255,180,72,0.8)');
  plotSeries('best',  '#00d9ff');

  // Labels
  chartCtx.fillStyle = '#8590b0';
  chartCtx.font = '10px Inter, sans-serif';
  chartCtx.fillText(`max ${Math.round(maxFit)}`, 4, 14);
  chartCtx.fillText('0', 4, H - 10);
  chartCtx.fillText(`gen ${history[history.length-1].gen}`, W - 40, H - 10);

  // Legend
  chartCtx.fillStyle = '#00d9ff';
  chartCtx.fillRect(W - 160, 8, 8, 8);
  chartCtx.fillStyle = '#8590b0';
  chartCtx.fillText('best', W - 148, 16);
  chartCtx.fillStyle = '#ffb648';
  chartCtx.fillRect(W - 110, 8, 8, 8);
  chartCtx.fillStyle = '#8590b0';
  chartCtx.fillText('avg', W - 98, 16);
  chartCtx.fillStyle = '#ff5470';
  chartCtx.fillRect(W - 70, 8, 8, 8);
  chartCtx.fillStyle = '#8590b0';
  chartCtx.fillText('worst', W - 58, 16);
}

/* ---------- Game State ---------- */
let level = attachLevelHelpers(generateLevel(settings.seed, settings.levelLen, settings.difficulty));
let pop = new Population(level);
let running = false;
let speedMult = 1;
let lastTime = 0;
let genStartTime = 0;

/* ---------- UI Binding ---------- */
const $ = id => document.getElementById(id);

function bindRange(id, valId, fn, formatter) {
  const el = $(id), v = $(valId);
  const update = () => {
    const val = +el.value;
    fn(val);
    if (v) v.textContent = formatter ? formatter(val) : val;
  };
  el.addEventListener('input', update);
  update();
}

function bindCheckbox(id, key) {
  const el = $(id);
  el.addEventListener('change', () => { settings[key] = el.checked; });
  el.checked = settings[key];
}

function bindAll() {
  // Evolution
  bindRange('popSize', 'vPop', v => settings.popSize = v);
  bindRange('mutRate', 'vMutRate', v => settings.mutRate = v / 100, v => v + '%');
  bindRange('mutAmt', 'vMutAmt', v => settings.mutAmt = v / 100, v => (v / 100).toFixed(2));
  bindRange('elite', 'vElite', v => settings.elitism = v);
  bindRange('selPct', 'vSel', v => settings.selPct = v / 100, v => v + '%');
  bindCheckbox('crossover', 'crossover');
  // NN
  $('hidden').addEventListener('change', e => {
    const parts = e.target.value.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0);
    if (parts.length === 0) { e.target.value = settings.hidden.join(','); return; }
    settings.hidden = parts;
    toast('Hidden layers updated — applies to new brains only.');
  });
  $('hidden').value = settings.hidden.join(',');
  bindRange('rayCount', 'vRays', v => settings.rayCount = v);
  bindRange('rayRange', 'vRange', v => settings.rayRange = v, v => v + 'px');
  bindCheckbox('autorun', 'autorun');
  bindCheckbox('doubleJump', 'doubleJump');
  // Physics
  bindRange('gravity', 'vGrav', v => settings.gravity = v / 100, v => (v / 100).toFixed(2));
  bindRange('jump', 'vJump', v => settings.jumpForce = v / 10, v => (v / 10).toFixed(1));
  bindRange('runSpeed', 'vRun', v => settings.runSpeed = v / 10, v => (v / 10).toFixed(1));
  bindRange('timeout', 'vTimeout', v => settings.timeout = v, v => v + 's');
  // Level
  $('seed').addEventListener('change', e => {
    const s = parseInt(e.target.value, 10);
    if (isNaN(s)) { e.target.value = settings.seed; return; }
    settings.seed = s;
  });
  bindRange('levelLen', 'vLen', v => settings.levelLen = v, v => v + 'px');
  bindRange('difficulty', 'vDiff', v => settings.difficulty = v, v => ['Easy','Medium','Hard','Brutal'][v]);
  $('btnNewLevel').addEventListener('click', () => {
    settings.seed = Math.floor(Math.random() * 100000);
    $('seed').value = settings.seed;
    regenerateLevel();
    toast('New level generated (seed ' + settings.seed + ')');
  });
  // Viz
  bindCheckbox('showRays', 'showRays');
  bindCheckbox('showAll', 'showAll');
  bindCheckbox('showDead', 'showDead');
  bindCheckbox('followCam', 'followCam');

  // Top controls
  $('btnPlay').addEventListener('click', () => {
    running = !running;
    $('btnPlay').textContent = running ? '⏸ Pause' : '▶ Start Training';
    $('btnPlay').classList.toggle('primary', !running);
    if (running) { lastTime = performance.now(); genStartTime = performance.now(); }
  });
  $('btnSkip').addEventListener('click', () => {
    if (!running) { running = true; $('btnPlay').textContent = '⏸ Pause'; }
    forceEvolve();
  });
  $('btnReset').addEventListener('click', () => {
    if (!confirm('Reset all training? This clears the population and generation history.')) return;
    fullReset();
  });

  // Speed presets
  document.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.chip').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      speedMult = +btn.dataset.speed;
    });
  });

  // Tabs
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      $('tab-' + tab.dataset.tab).classList.add('active');
    });
  });

  // Save / load brain
  $('btnSave').addEventListener('click', saveBestBrain);
  $('btnLoad').addEventListener('change', loadBrainFromFile);
  $('btnExport').addEventListener('click', exportSettings);
  $('btnImport').addEventListener('change', importSettingsFromFile);
  $('btnUseBest').addEventListener('click', () => {
    if (pop.bestEver) {
      pop.injectBest();
      toast('Best brain injected into the population.');
    }
  });
}

function regenerateLevel() {
  level = attachLevelHelpers(generateLevel(settings.seed, settings.levelLen, settings.difficulty));
  pop = new Population(level);
  genStartTime = performance.now();
}

function fullReset() {
  pop = new Population(level);
  genStartTime = performance.now();
}

function forceEvolve() {
  pop.evolve();
  genStartTime = performance.now();
  updateGenLog();
  updateBrainsTab();
}

/* ---------- Save / Load ---------- */
function saveBestBrain() {
  if (!pop.bestEver) { toast('No best brain yet — train one first.'); return; }
  const data = {
    type: 'NeuroParkour-Brain',
    brain: pop.bestEver.brain.serialize(),
    fitness: pop.bestEver.fitness,
    generation: pop.bestEver.generation,
    settings: settings,
  };
  downloadJSON(data, `neuroparkour-brain-gen${pop.bestEver.generation}.json`);
  toast('Best brain saved.');
}
function loadBrainFromFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    try {
      const data = JSON.parse(ev.target.result);
      if (data.type !== 'NeuroParkour-Brain' || !data.brain) throw new Error('Invalid file');
      const brain = NeuralNetwork.deserialize(data.brain);
      pop.bestEver = {
        brain,
        fitness: data.fitness || 0,
        generation: data.generation || 0,
        maxX: 0,
        reachedGoal: false,
      };
      $('btnUseBest').disabled = false;
      updateBrainsTab();
      toast('Brain loaded — click "Load Best into Population" to use it.');
    } catch (err) {
      toast('Error loading brain: ' + err.message);
    }
  };
  reader.readAsText(file);
  e.target.value = '';
}
function exportSettings() {
  downloadJSON({ type: 'NeuroParkour-Settings', settings }, 'neuroparkour-settings.json');
  toast('Settings exported.');
}
function importSettingsFromFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    try {
      const data = JSON.parse(ev.target.result);
      if (data.type !== 'NeuroParkour-Settings' || !data.settings) throw new Error('Invalid file');
      Object.assign(settings, data.settings);
      // Re-bind UI
      syncUIFromSettings();
      regenerateLevel();
      toast('Settings imported.');
    } catch (err) {
      toast('Error importing settings: ' + err.message);
    }
  };
  reader.readAsText(file);
  e.target.value = '';
}
function downloadJSON(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function syncUIFromSettings() {
  $('popSize').value = settings.popSize; $('vPop').textContent = settings.popSize;
  $('mutRate').value = settings.mutRate * 100; $('vMutRate').textContent = (settings.mutRate * 100) + '%';
  $('mutAmt').value = settings.mutAmt * 100; $('vMutAmt').textContent = settings.mutAmt.toFixed(2);
  $('elite').value = settings.elitism; $('vElite').textContent = settings.elitism;
  $('selPct').value = settings.selPct * 100; $('vSel').textContent = (settings.selPct * 100) + '%';
  $('crossover').checked = settings.crossover;
  $('hidden').value = settings.hidden.join(',');
  $('rayCount').value = settings.rayCount; $('vRays').textContent = settings.rayCount;
  $('rayRange').value = settings.rayRange; $('vRange').textContent = settings.rayRange + 'px';
  $('autorun').checked = settings.autorun;
  $('doubleJump').checked = settings.doubleJump;
  $('gravity').value = settings.gravity * 100; $('vGrav').textContent = settings.gravity.toFixed(2);
  $('jump').value = settings.jumpForce * 10; $('vJump').textContent = settings.jumpForce.toFixed(1);
  $('runSpeed').value = settings.runSpeed * 10; $('vRun').textContent = settings.runSpeed.toFixed(1);
  $('timeout').value = settings.timeout; $('vTimeout').textContent = settings.timeout + 's';
  $('seed').value = settings.seed;
  $('levelLen').value = settings.levelLen; $('vLen').textContent = settings.levelLen + 'px';
  $('difficulty').value = settings.difficulty; $('vDiff').textContent = ['Easy','Medium','Hard','Brutal'][settings.difficulty];
  $('showRays').checked = settings.showRays;
  $('showAll').checked = settings.showAll;
  $('showDead').checked = settings.showDead;
  $('followCam').checked = settings.followCam;
}

/* ---------- UI Updates ---------- */
function updateStats() {
  $('statGen').textContent = pop.generation;
  $('statAlive').textContent = pop.aliveCount();
  $('statBest').textContent = Math.round(pop.bestEver ? pop.bestEver.maxX || pop.bestEver.fitness : 0);
  const leader = pop.leader();
  $('statLeader').textContent = leader ? Math.round(leader.maxX) : 0;
  $('statTime').textContent = ((performance.now() - genStartTime) / 1000).toFixed(1) + 's';
  // Progress bar
  const leaderX = leader ? leader.maxX : 0;
  const pct = clamp(leaderX / level.goalX * 100, 0, 100);
  $('progressFill').style.width = pct + '%';
  // Stagnation indicator (color shifts to warn as stagnation grows)
  const stagEl = $('statStag');
  if (stagEl) {
    stagEl.textContent = pop.stagnation + (pop.stagnationBoost > 1 ? ' (' + pop.stagnationBoost + '×)' : '');
    stagEl.style.color =
      pop.stagnation > 20 ? '#ff5470' :
      pop.stagnation > 10 ? '#ffb648' : '#00d9ff';
  }
}

function updateGenLog() {
  const log = $('genLog');
  log.innerHTML = '';
  const recent = pop.history.slice(-20).reverse();
  for (const h of recent) {
    const div = document.createElement('div');
    div.className = 'gen-log-entry';
    div.innerHTML = `<span class="gen-num">Gen ${h.gen}</span> · best <span class="best">${Math.round(h.best)}</span> · avg ${Math.round(h.avg)} · worst ${Math.round(h.worst)}`;
    log.appendChild(div);
  }
}
function updateBrainsTab() {
  const info = $('bestBrainInfo');
  if (pop.bestEver) {
    info.innerHTML = `
      <div><strong>Generation:</strong> ${pop.bestEver.generation}</div>
      <div><strong>Fitness:</strong> ${Math.round(pop.bestEver.fitness)}</div>
      <div><strong>Max distance:</strong> ${Math.round(pop.bestEver.maxX || 0)}px</div>
      <div><strong>Reached goal:</strong> ${pop.bestEver.reachedGoal ? 'Yes 🏆' : 'No'}</div>
      <div><strong>Network:</strong> ${pop.bestEver.brain.sizes.join(' → ')}</div>
    `;
    $('btnUseBest').disabled = false;
  } else {
    info.textContent = 'No best brain yet — train one!';
    $('btnUseBest').disabled = true;
  }
  // Snapshot
  const snap = $('popSnapshot');
  snap.innerHTML = '';
  const items = [
    ['Population', pop.agents.length],
    ['Alive', pop.aliveCount()],
    ['Generation', pop.generation],
    ['Network', pop.agents[0] ? pop.agents[0].brain.sizes.join('-') : '-'],
    ['Best ever', pop.bestEver ? Math.round(pop.bestEver.fitness) : 0],
    ['History len', pop.history.length],
  ];
  for (const [k, v] of items) {
    const div = document.createElement('div');
    div.className = 'snapshot-item';
    div.innerHTML = `<span class="k">${k}</span><span class="v">${v}</span>`;
    snap.appendChild(div);
  }
}

/* ---------- Toast ---------- */
let toastTimer = null;
function toast(msg) {
  let el = document.querySelector('.toast');
  if (!el) {
    el = document.createElement('div');
    el.className = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2400);
}

/* ---------- Main Loop ---------- */
function loop(now) {
  requestAnimationFrame(loop);
  if (!running) {
    // Still render the static scene
    render();
    return;
  }
  const realDt = Math.min(0.05, (now - lastTime) / 1000);
  lastTime = now;

  // Run multiple simulation steps based on speed multiplier
  const steps = speedMult >= 30 ? 30 : speedMult;
  const stepDt = realDt;
  let evolveTriggered = false;
  for (let i = 0; i < steps; i++) {
    pop.step(stepDt);
    if (pop.isAllDead()) {
      pop.evolve();
      genStartTime = performance.now();
      updateGenLog();
      updateBrainsTab();
      evolveTriggered = true;
      break;
    }
  }

  // If turbo and not yet all dead, run extra idle steps until time budget consumed
  if (speedMult >= 30 && !evolveTriggered) {
    const budget = 12; // ms
    const t0 = performance.now();
    while (performance.now() - t0 < budget) {
      pop.step(stepDt);
      if (pop.isAllDead()) {
        pop.evolve();
        genStartTime = performance.now();
        updateGenLog();
        updateBrainsTab();
        break;
      }
    }
  }

  render();
}

function render() {
  const leader = pop.leader();
  // Camera
  if (settings.followCam && leader) {
    const targetX = leader.x - canvas.clientWidth / 2.5;
    camera.x = lerp(camera.x, targetX, 0.1);
    camera.x = clamp(camera.x, 0, level.width - canvas.clientWidth);
  }
  drawLevel(level, camera);
  drawAgents(pop, leader);
  drawChart(pop.history);
  updateStats();
}

/* ---------- Init ---------- */
function init() {
  bindAll();
  syncUIFromSettings();
  resizeCanvas();
  // Initial camera position
  camera.x = 0;
  // Initial render
  updateBrainsTab();
  updateGenLog();
  requestAnimationFrame(loop);
}

init();
