import { mulberry32 } from './prng.js';
import { Particle } from './particle.js';
import { SpatialGrid } from './spatial-grid.js';
import { DEFAULT_TYPES, PRESETS } from './constants.js';
import { SoundEngine } from './sound.js';

/**
 * Convert a hex color string to {r,g,b} (0-255). Falls back to white
 * for unparseable input.
 */
function hexToRgb(hex) {
  let h = (hex || '').replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  const n = parseInt(h, 16);
  if (isNaN(n)) return { r: 255, g: 255, b: 255 };
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

/**
 * Simulation — main loop, physics, and rendering.
 * Manages particles, camera (zoom/pan), fixed-timestep integration,
 * and renders everything to a world-space canvas that the camera
 * zooms/pan around.
 */
export class Simulation {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // Config
    this.totalParticles = 1200;
    this.interactionRadius = 80;
    this.maxSpeed = 1.5;
    this.damping = 0.96;
    this.seed = Math.floor(Math.random() * 99999);
    this.trail = 30;
    this.glow = true;
    this.glowSize = 12;
    this.glowIntensity = 0.05;
    this.bgColor = '#050508';
    this.showVectors = false;
    this.showGrid = false;
    this.wrap = false;

    // Life cycle (reproduction & death) — see lifeStep()
    this.lifeEnabled = true;
    this.energyDecay = 0.03;   // energy lost per second (metabolism)
    this.collisionCost = 0.1;  // energy lost per second at full contact
    this.feedRate = 0.06;      // energy gained per second per unit of attraction
    this.reproNeighbors = 3;   // min neighbors within radius to reproduce
    this.reproEnergy = 0.8;    // min energy to split
    this.maxParticles = 5000;  // population cap
    this.reproCooldown = 5;    // seconds before a particle can split again
    this.births = 0;
    this.deaths = 0;
    this.deathFx = [];

    // Sound
    this.soundEnabled = false;

    // Camera (zoom + pan)
    this.viewX = 0;
    this.viewY = 0;
    this.zoom = 1;
    this.maxZoom = 20;

    // Particle types
    this.types = JSON.parse(JSON.stringify(DEFAULT_TYPES));

    // Interaction matrix (NxN)
    this.numTypes = this.types.length;
    this.matrix = PRESETS.swirl(this.numTypes);

    // Runtime
    this.particles = [];
    this.grid = new SpatialGrid(this.interactionRadius);
    this.running = true;
    this.rng = mulberry32(this.seed);

    // Fixed timestep + adaptive timescale (see loop()).
    // The render is locked to targetFps. Physics integrates in fixedDt
    // sub-steps; when the machine can't keep up with real time, `timescale`
    // is lowered so the simulation slows down instead of the frame rate
    // dropping below 60fps.
    this.fixedDt = 1 / 60;
    this.targetFps = 60;
    this.timescale = 1;          // sim-speed multiplier (1 = real time)
    this.minTimescale = 0.25;    // never slower than 1/4 real-time
    this.maxSteps = 8;           // hard cap on physics steps per render frame
    this.simAccum = 0;           // leftover sim-time awaiting fixed steps
    this._stepCost = 0;          // rolling average ms per physics step
    this.nextFrameTime = 0;      // render "phase" — gate for the 60fps lock

    // FPS tracking
    this.frameCount = 0;
    this.fpsTime = 0;
    this.fps = 0;

    // World canvas — offscreen canvas at viewport size that holds
    // particles + trails in world coordinates. The main canvas draws
    // this canvas through the camera transform.
    this.worldCanvas = document.createElement('canvas');
    this.worldCtx = this.worldCanvas.getContext('2d');

    // Sound engine for interaction feedback
    this.sound = new SoundEngine(this);

    // Pre-render one soft radial sprite per particle color for the bloom
    // effect (drawn cheaply with additive compositing at render time).
    this.buildGlowSprites();

    this.resize();
    this.initParticles();
    this.bindCamera();
  }

  /** Resize canvases to match container. */
  resize() {
    const wrap = this.canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    this.w = wrap.clientWidth;
    this.h = wrap.clientHeight;

    this.canvas.width = this.w * dpr;
    this.canvas.height = this.h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // World canvas — same logical size as viewport
    this.worldCanvas.width = this.w * dpr;
    this.worldCanvas.height = this.h * dpr;
    this.worldCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Clear both canvases on resize (trails rebuild naturally)
    this.ctx.clearRect(0, 0, this.w, this.h);
    this.worldCtx.clearRect(0, 0, this.w, this.h);
  }

  /* ---- Camera (zoom + pan) ---- */
  bindCamera() {
    const canvas = this.canvas;

    // Scroll wheel zoom (centered on mouse position)
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const wx = mx / this.zoom + this.viewX;
      const wy = my / this.zoom + this.viewY;

      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      this.zoom = Math.max(0.5, Math.min(this.maxZoom, this.zoom * factor));

      this.viewX = wx - mx / this.zoom;
      this.viewY = wy - my / this.zoom;
    }, { passive: false });

    // Mouse drag pan (screen-space: 1:1 cursor movement)
    let dragging = false;
    let dragStartX, dragStartY;
    let viewStartX, viewStartY;

    canvas.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      dragging = true;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      viewStartX = this.viewX;
      viewStartY = this.viewY;
      canvas.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      this.viewX = viewStartX - dx;
      this.viewY = viewStartY - dy;
    });

    window.addEventListener('mouseup', () => {
      if (dragging) {
        dragging = false;
        canvas.style.cursor = '';
      }
    });

    // Reset view with double-click
    canvas.addEventListener('dblclick', () => {
      this.zoom = 1;
      this.viewX = 0;
      this.viewY = 0;
    });
  }

  /* ---- Particle management ---- */
  initParticles() {
    this.rng = mulberry32(this.seed);
    this.particles = [];
    this.births = 0;
    this.deaths = 0;
    this.deathFx.length = 0;

    // Compute per-type counts from percentages
    const counts = [];
    let totalPct = 0;
    for (const t of this.types) totalPct += t.countPct;
    const norm = totalPct / 100;

    for (let i = 0; i < this.types.length; i++) {
      const t = this.types[i];
      const cnt = Math.max(1, Math.round((t.countPct / totalPct) * this.totalParticles));
      counts.push(cnt);
    }

    // Adjust to match total exactly
    const diff = this.totalParticles - counts.reduce((a, b) => a + b, 0);
    if (diff !== 0) {
      counts[0] = Math.max(1, counts[0] + diff);
    }

    for (let t = 0; t < this.types.length; t++) {
      for (let i = 0; i < counts[t]; i++) {
        const x = this.rng() * this.w;
        const y = this.rng() * this.h;
        const angle = this.rng() * Math.PI * 2;
        const speed = this.rng() * 0.5;
        const p = new Particle(x, y, t, this.particles.length);
        p.vx = Math.cos(angle) * speed;
        p.vy = Math.sin(angle) * speed;
        this.particles.push(p);
      }
    }
  }

  adjustParticleCount() {
    const current = this.particles.length;
    if (current === this.totalParticles) return;

    if (this.totalParticles > current) {
      for (let i = 0; i < this.totalParticles - current; i++) {
        const t = Math.floor(this.rng() * this.types.length);
        const p = new Particle(this.rng() * this.w, this.rng() * this.h, t, this.particles.length);
        const angle = this.rng() * Math.PI * 2;
        const speed = this.rng() * 0.5;
        p.vx = Math.cos(angle) * speed;
        p.vy = Math.sin(angle) * speed;
        this.particles.push(p);
      }
    } else {
      this.particles.length = this.totalParticles;
    }
  }

  /* ---- Physics step ---- */
  step() {
    const dt = this.fixedDt;
    const radius = this.interactionRadius;
    const maxSpeed = this.maxSpeed;
    const damping = this.damping;
    const maxForce = maxSpeed * 2;

    this.grid.cellSize = radius;
    this.grid.build(this.particles, this.w, this.h);

    const particles = this.particles;
    const n = particles.length;
    const matrix = this.matrix;
    const numTypes = this.numTypes;

    // Life-cycle accumulators (per particle) — only active when life is on
    const lifeOn = this.lifeEnabled;
    if (lifeOn) {
      if (!this._lifeBuf || this._lifeBuf.length < n) {
        this._lifeBuf = new Float32Array(n);
        this._feedBuf = new Float32Array(n);
        this._contactBuf = new Float32Array(n);
      }
      this._lifeBuf.fill(0, 0, n);
      this._feedBuf.fill(0, 0, n);
      this._contactBuf.fill(0, 0, n);
    }
    const nCount = lifeOn ? this._lifeBuf : null;
    const feed = lifeOn ? this._feedBuf : null;
    const contact = lifeOn ? this._contactBuf : null;

    for (let i = 0; i < n; i++) {
      const p = particles[i];
      let fx = 0, fy = 0;

      const neighbors = this.grid.getNeighbors(p, radius);
      for (let j = 0; j < neighbors.length; j++) {
        const q = neighbors[j];
        if (q.type >= numTypes) continue;

        let dx = q.x - p.x;
        let dy = q.y - p.y;

        if (this.wrap) {
          if (dx > this.w * 0.5) dx -= this.w;
          else if (dx < -this.w * 0.5) dx += this.w;
          if (dy > this.h * 0.5) dy -= this.h;
          else if (dy < -this.h * 0.5) dy += this.h;
        }

        const dSq = dx * dx + dy * dy;
        if (dSq >= radius * radius || dSq < 0.01) continue;

        const d = Math.sqrt(dSq);
        const strength = matrix[p.type]?.[q.type] ?? 0;

        const f = strength * (1 - d / radius);

        fx += (dx / d) * f;
        fy += (dy / d) * f;

        // Life-cycle accumulators: neighbor count, feeding (attraction),
        // and close-contact cost
        if (lifeOn) {
          nCount[i]++;
          if (strength > 0) feed[i] += f;
          if (d < 5) contact[i] += (5 - d) / 5;
        }

        // Separation — push apart when very close
        if (d < 5) {
          const sep = (5 - d) / 5 * 0.5;
          fx -= (dx / d) * sep * 2;
          fy -= (dy / d) * sep * 2;
        }
      }

      p.vx += fx * dt;
      p.vy += fy * dt;

      const spd = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
      if (spd > maxSpeed) {
        const inv = maxSpeed / spd;
        p.vx *= inv;
        p.vy *= inv;
      }

      p.vx *= damping;
      p.vy *= damping;

      p.x += p.vx;
      p.y += p.vy;

      this.wrapTorus(p);
    }

    if (lifeOn) {
      this.lifeStep(dt, nCount, feed, contact);
      this.updateDeathFx(dt);
    }
  }

  /* ---- Torus wrapping (corner-aware) ---- */
  /**
   * Wrap a particle back into the world after it moved out of bounds.
   *
   * Straight exits: crossing an edge teleports the particle to the
   * directly opposite point (same distance from the opposite edge, same
   * direction of travel), e.g. left edge -> right edge.
   *
   * Corner exits: if a particle crosses one edge while it is within a
   * few frames of travel of the adjacent edge (i.e. it is really
   * leaving through a corner), the adjacent axis is mirrored to the
   * opposite side in the same step. That makes a corner exit a single
   * teleport to the opposite corner with the exit angle preserved,
   * instead of two separate axis-aligned jumps a few frames apart.
   *
   * prevX/prevY are transformed with the same mapping so the trail
   * segment stays short and is drawn at the new location.
   */
  wrapTorus(p) {
    const w = this.w, h = this.h;
    const MIN_LA = 2;  // px — minimum corner look-ahead distance
    const MAX_LA = 10; // px — maximum corner look-ahead distance
    const FRAMES = 4;  // travel within this many frames = "at the corner"

    let wrappedX = false, wrappedY = false;

    if (p.x < 0) { p.x += w; p.prevX += w; wrappedX = true; }
    else if (p.x >= w) { p.x -= w; p.prevX -= w; wrappedX = true; }
    if (p.y < 0) { p.y += h; p.prevY += h; wrappedY = true; }
    else if (p.y >= h) { p.y -= h; p.prevY -= h; wrappedY = true; }

    // Exactly one axis crossed an edge this frame (an exit event). If
    // the particle is also about to cross the adjacent edge, mirror it
    // to the opposite side right now: one corner-to-corner teleport.
    if (wrappedX !== wrappedY) {
      if (p.vx > 0) {
        const la = Math.min(MAX_LA, Math.max(MIN_LA, FRAMES * p.vx));
        if (p.x > w - la) { p.x = w - p.x; p.prevX = w - p.prevX; }
      } else if (p.vx < 0) {
        const la = Math.min(MAX_LA, Math.max(MIN_LA, FRAMES * -p.vx));
        if (p.x < la) { p.x = w - p.x; p.prevX = w - p.prevX; }
      }
      if (p.vy > 0) {
        const la = Math.min(MAX_LA, Math.max(MIN_LA, FRAMES * p.vy));
        if (p.y > h - la) { p.y = h - p.y; p.prevY = h - p.prevY; }
      } else if (p.vy < 0) {
        const la = Math.min(MAX_LA, Math.max(MIN_LA, FRAMES * -p.vy));
        if (p.y < la) { p.y = h - p.y; p.prevY = h - p.prevY; }
      }
    }
  }

  /* ---- Life cycle: energy, death, reproduction ---- */
  /**
   * Life-cycle step, modeled on two well-known particle-life variants:
   *
   * 1. Energy/health — as in najarro.science/pl's "Energy" mode and
   *    jkh2/Primordial-Sim:
   *      E += feedRate · Σ(attractive interactions)
   *           − energyDecay − collisionCost · Σ(close contacts)   [per sec]
   *    A particle dies (is removed) when E reaches 0.
   *
   * 2. Reproduction — as in the Primordial Particle System (Schmickl &
   *    Stefanec, A-Life Lab Graz): when a particle has enough neighbors
   *    within the interaction radius and enough energy, it splits — the
   *    child spawns between the parent and its neighbor centroid and the
   *    two share the parent's energy 50/50. Population is capped.
   */
  lifeStep(dt, nCount, feed, contact) {
    const particles = this.particles;
    const decay = this.energyDecay;
    const collision = this.collisionCost;
    const feedRate = this.feedRate;
    const reproN = this.reproNeighbors;
    const reproE = this.reproEnergy;

    const survivors = [];
    const candidates = [];

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.age += dt;
      if (p.reproCooldown > 0) p.reproCooldown -= dt;

      p.energy += (feedRate * feed[i] - decay - collision * contact[i]) * dt;
      if (p.energy > 1) p.energy = 1;

      if (p.energy <= 0) {
        this.deaths++;
        this.spawnDeathFx(p);
        if (this.soundEnabled) this.sound.playDeath();
        continue; // starved — removed
      }

      if (nCount[i] >= reproN && p.energy >= reproE && p.reproCooldown <= 0) {
        candidates.push(p);
      }
      survivors.push(p);
    }

    this.particles = survivors;

    // Splits — limited by the population cap
    let budget = this.maxParticles - survivors.length;
    if (budget > 0) {
      for (let c = 0; c < candidates.length && budget > 0; c++) {
        if (this.spawnChild(candidates[c])) {
          this.births++;
          budget--;
          if (this.soundEnabled) this.sound.playBirth();
        }
      }
    }
  }

  /**
   * PPS-style split: spawn a same-type child between the parent and its
   * neighbor centroid; parent and child each keep 50% of the parent's
   * energy. Returns true if a child was created.
   */
  spawnChild(parent) {
    const radius = this.interactionRadius;
    const neighbors = this.grid.getNeighbors(parent, radius);

    // Centroid of the parent's neighbors within the interaction radius
    let cx = 0, cy = 0, cnt = 0;
    for (let j = 0; j < neighbors.length; j++) {
      const q = neighbors[j];
      if (q.type >= this.numTypes) continue;
      let dx = q.x - parent.x;
      let dy = q.y - parent.y;
      if (this.wrap) {
        if (dx > this.w * 0.5) dx -= this.w;
        else if (dx < -this.w * 0.5) dx += this.w;
        if (dy > this.h * 0.5) dy -= this.h;
        else if (dy < -this.h * 0.5) dy += this.h;
      }
      const dSq = dx * dx + dy * dy;
      if (dSq >= radius * radius || dSq < 0.01) continue;
      cx += dx;
      cy += dy;
      cnt++;
    }

    let ox, oy;
    if (cnt > 0) {
      // Halfway between the parent and its neighbor centroid (the child
      // splits off *inside* the cluster, like PPS)
      ox = cx / cnt * 0.5;
      oy = cy / cnt * 0.5;
    } else {
      const a = this.rng() * Math.PI * 2;
      ox = Math.cos(a) * 10;
      oy = Math.sin(a) * 10;
    }
    // Jitter so parent and child never perfectly overlap
    ox += (this.rng() - 0.5) * 6;
    oy += (this.rng() - 0.5) * 6;

    const child = new Particle(parent.x + ox, parent.y + oy, parent.type, this.particles.length);
    child.energy = parent.energy * 0.5;
    parent.energy *= 0.5;
    parent.reproCooldown = this.reproCooldown;
    child.reproCooldown = this.reproCooldown;
    const a = this.rng() * Math.PI * 2;
    const s = this.rng() * 0.5;
    child.vx = Math.cos(a) * s;
    child.vy = Math.sin(a) * s;
    this.wrapTorus(child);
    this.particles.push(child);
    return true;
  }

  /** Spawn a small fading burst where a particle died. */
  spawnDeathFx(p) {
    if (this.deathFx.length >= 400) return;
    const n = 2;
    for (let i = 0; i < n; i++) {
      const a = this.rng() * Math.PI * 2;
      const s = 0.2 + this.rng() * 0.6;
      this.deathFx.push({
        x: p.x, y: p.y,
        vx: Math.cos(a) * s, vy: Math.sin(a) * s,
        life: 0.8 + this.rng() * 0.2,
        type: p.type,
      });
    }
  }

  /** Age and drift the death bursts; drop the expired ones. */
  updateDeathFx(dt) {
    const fx = this.deathFx;
    for (let i = fx.length - 1; i >= 0; i--) {
      const f = fx[i];
      f.life -= dt * 1.6;
      if (f.life <= 0) { fx.splice(i, 1); continue; }
      f.x += f.vx;
      f.y += f.vy;
      f.vx *= 0.94;
      f.vy *= 0.94;
    }
  }

  /* ---- Glow sprites ---- */
  /** Pre-render a soft radial "glow" sprite for a given color. */
  makeGlowSprite(color) {
    const R = 36; // sprite covers radius R; gradient fades to transparent at edge
    const c = document.createElement('canvas');
    c.width = c.height = R * 2;
    const g = c.getContext('2d');
    const { r: cr, g: cg, b: cb } = hexToRgb(color);
    const grad = g.createRadialGradient(R, R, 0, R, R, R);
    grad.addColorStop(0.0, `rgba(${cr},${cg},${cb},0.90)`);
    grad.addColorStop(0.2, `rgba(${cr},${cg},${cb},0.55)`);
    grad.addColorStop(0.5, `rgba(${cr},${cg},${cb},0.16)`);
    grad.addColorStop(1.0, `rgba(${cr},${cg},${cb},0)`);
    g.fillStyle = grad;
    g.fillRect(0, 0, R * 2, R * 2);
    return c;
  }

  /** Build one glow sprite per particle type (rebuild when colors change). */
  buildGlowSprites() {
    this.glowSprites = this.types.map(t => this.makeGlowSprite(t.color));
  }

  /* ---- Render ---- */
  render() {
    const ctx = this.ctx;
    const worldCtx = this.worldCtx;
    const trail = this.trail;
    const zoom = this.zoom;
    const viewX = this.viewX;
    const viewY = this.viewY;

    if (trail > 0) {
      // Fade everything on world canvas via destination-out (gradually removes old trails)
      const alpha = Math.max(0.02, 1 - trail / 32);
      worldCtx.globalCompositeOperation = 'destination-out';
      worldCtx.fillStyle = `rgba(0,0,0,${alpha})`;
      worldCtx.fillRect(0, 0, this.w, this.h);
      worldCtx.globalCompositeOperation = 'source-over';

      // Draw new trail lines in world space
      for (let i = 0; i < this.particles.length; i++) {
        const p = this.particles[i];
        const t = this.types[p.type];
        if (!t) continue;

        worldCtx.beginPath();
        worldCtx.moveTo(p.prevX, p.prevY);
        worldCtx.lineTo(p.x, p.y);
        worldCtx.strokeStyle = t.color;
        worldCtx.lineWidth = t.size;
        worldCtx.lineCap = 'round';
        worldCtx.stroke();
      }
    } else {
      // No trails: clear world canvas so old trail content doesn't linger
      worldCtx.clearRect(0, 0, this.w, this.h);
    }

    // Draw particles on top of trails (world space)
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      const t = this.types[p.type];
      if (!t) continue;

      worldCtx.beginPath();
      worldCtx.arc(p.x, p.y, t.size, 0, Math.PI * 2);
      worldCtx.fillStyle = t.color;
      worldCtx.fill();

      if (this.showVectors) {
        worldCtx.beginPath();
        worldCtx.moveTo(p.x, p.y);
        worldCtx.lineTo(p.x + p.vx * 8, p.y + p.vy * 8);
        worldCtx.strokeStyle = 'rgba(255,255,255,0.25)';
        worldCtx.lineWidth = 0.5;
        worldCtx.stroke();
      }
    }

    // --- Main canvas: solid background + world canvas through camera transform ---
    ctx.fillStyle = this.bgColor;
    ctx.fillRect(0, 0, this.w, this.h);

    ctx.save();
    ctx.translate(-viewX * zoom, -viewY * zoom);
    ctx.scale(zoom, zoom);
    // Draw the world canvas into its LOGICAL size (w × h), not its bitmap
    // size (w*dpr × h*dpr). This keeps world-logical units 1:1 with the main
    // canvas' logical coordinates at any devicePixelRatio, so screen-space
    // effects (the glow below) stay aligned with the particles. Without the
    // explicit size, retina displays render the world at dpr× scale and the
    // glow ends up offset toward the top-left.
    ctx.drawImage(this.worldCanvas, 0, 0, this.w, this.h);
    ctx.restore();

    // --- Glow / bloom pass (additive, screen space) ---
    // Drawn on the MAIN canvas, which is cleared fresh every frame, *after*
    // the trails. So the bloom never accumulates energy in the trail — it is
    // a per-frame bloom layered on top of the accumulated trails. Pre-rendered
    // per-color sprites are blitted with additive ("lighter") compositing so
    // overlapping particles bloom into hot bright spots. Rendered in screen
    // space at native resolution, so it stays crisp at every zoom level.
    if (this.glow && this.glowIntensity > 0) {
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = this.glowIntensity;
      for (let i = 0; i < this.particles.length; i++) {
        const p = this.particles[i];
        const t = this.types[p.type];
        if (!t) continue;
        let sprite = this.glowSprites[p.type];
        if (!sprite) sprite = this.glowSprites[p.type] = this.makeGlowSprite(t.color);
        const sx = (p.x - viewX) * zoom;
        const sy = (p.y - viewY) * zoom;
        const r = t.size * this.glowSize * zoom;
        // Skip particles fully outside the viewport
        if (sx + r < 0 || sx - r > this.w || sy + r < 0 || sy - r > this.h) continue;
        ctx.drawImage(sprite, sx - r, sy - r, r * 2, r * 2);
      }
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = 1;
    }

    // --- Death bursts (life cycle) ---
    if (this.lifeEnabled && this.deathFx.length > 0) {
      ctx.globalCompositeOperation = 'lighter';
      for (let i = 0; i < this.deathFx.length; i++) {
        const f = this.deathFx[i];
        const t = this.types[f.type];
        if (!t) continue;
        let sprite = this.glowSprites[f.type];
        if (!sprite) sprite = this.glowSprites[f.type] = this.makeGlowSprite(t.color);
        const r = t.size * this.glowSize * zoom * f.life;
        const sx = (f.x - viewX) * zoom;
        const sy = (f.y - viewY) * zoom;
        if (sx + r < 0 || sx - r > this.w || sy + r < 0 || sy - r > this.h) continue;
        ctx.globalAlpha = Math.min(0.6, f.life * 0.5);
        ctx.drawImage(sprite, sx - r, sy - r, r * 2, r * 2);
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';
    }

    // Draw spatial grid in screen space (fixed line width, doesn't scale with zoom)
    if (this.showGrid) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.lineWidth = 0.5;
      const cs = this.grid.cellSize;
      for (let x = 0; x <= this.w; x += cs) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, this.h);
        ctx.stroke();
      }
      for (let y = 0; y <= this.h; y += cs) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(this.w, y);
        ctx.stroke();
      }
      ctx.restore();
    }

    // Update previous positions for next frame's trail lines
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      p.prevX = p.x;
      p.prevY = p.y;
    }

    // Update sound engine
    this.sound.update();
  }

  /* ---- Main loop ----
   *
   * The render is pinned to 60 fps. Physics runs on a fixed timestep
   * (fixedDt) but the *rate* at which it advances is controlled by an
   * adaptive `timescale`. Each render frame measures how long a physics step
   * takes; if the cost threatens the 60fps budget, the timescale eases down
   * so the simulation slows instead of the frame rate dropping. When there's
   * headroom the timescale eases back up to real time.
   */
  loop(timestamp) {
    requestAnimationFrame((t) => this.loop(t));

    // ---- 60 fps render gate ----
    // rAF fires at the display's native rate (up to 144 Hz+); render only when
    // the target phase is due so the frame rate stays pinned to targetFps.
    // Re-anchor the phase after a big hitch so we never burst-catch-up.
    const interval = 1000 / this.targetFps;
    if (this.nextFrameTime === 0) this.nextFrameTime = timestamp;
    if (timestamp < this.nextFrameTime) return;
    if (timestamp - this.nextFrameTime > interval * 3) this.nextFrameTime = timestamp;
    this.nextFrameTime += interval;

    // FPS
    this.frameCount++;
    if (timestamp - this.fpsTime >= 500) {
      this.fps = Math.round(this.frameCount / ((timestamp - this.fpsTime) / 1000));
      this.frameCount = 0;
      this.fpsTime = timestamp;
      if (globalThis.ui) globalThis.ui.updateStats();
    }

    if (!this.running) return;

    // ---- Adaptive timescale: slow the sim instead of dropping the frame rate ----
    // One render frame has a fixed budget (interval ms). We track the rolling
    // cost of a single physics step and compute the fastest timescale whose
    // step still fits the budget (with ~20% headroom for rendering). As the
    // cost rises, timescale eases down (sim slows); as headroom returns it
    // eases back up to 1 (real time).
    const estStep = Math.max(this._stepCost, 0.05);
    const target = Math.max(this.minTimescale, Math.min(1, (interval * 0.8) / estStep));
    this.timescale += (target - this.timescale) * 0.15;

    // ---- Fixed-timestep physics ----
    // Each rendered frame advances the sim by one nominal frame scaled by the
    // current timescale, then runs one fixedDt step per accrued unit. At
    // timescale 1 this is exactly one step/frame (real time); a lower timescale
    // means fewer steps per second, so the sim slows while the render holds 60fps.
    this.simAccum = Math.min(this.simAccum + this.fixedDt * this.timescale, this.fixedDt * this.maxSteps);
    const t0 = performance.now();
    let steps = 0;
    while (this.simAccum >= this.fixedDt && steps < this.maxSteps) {
      this.step();
      this.simAccum -= this.fixedDt;
      steps++;
    }
    if (steps > 0) {
      const perStep = (performance.now() - t0) / steps;
      this._stepCost = this._stepCost ? this._stepCost * 0.8 + perStep * 0.2 : perStep;
    }

    this.render();
  }

  start() {
    this.nextFrameTime = 0;
    this.simAccum = 0;
    this._stepCost = 0;
    this.timescale = 1;
    requestAnimationFrame((t) => this.loop(t));
  }

  /* ---- Export / Import ---- */
  exportConfig() {
    return {
      totalParticles: this.totalParticles,
      interactionRadius: this.interactionRadius,
      maxSpeed: this.maxSpeed,
      damping: this.damping,
      seed: this.seed,
      trail: this.trail,
      glow: this.glow,
      glowSize: this.glowSize,
      glowIntensity: this.glowIntensity,
      bgColor: this.bgColor,
      showVectors: this.showVectors,
      showGrid: this.showGrid,
      types: this.types.map(t => ({ ...t })),
      matrix: this.matrix.map(r => [...r]),
      soundEnabled: this.soundEnabled ?? false,
      life: {
        enabled: this.lifeEnabled,
        energyDecay: this.energyDecay,
        collisionCost: this.collisionCost,
        feedRate: this.feedRate,
        reproNeighbors: this.reproNeighbors,
        reproEnergy: this.reproEnergy,
        maxParticles: this.maxParticles,
        reproCooldown: this.reproCooldown,
      },
    };
  }

  importConfig(cfg) {
    if (cfg.totalParticles !== undefined) this.totalParticles = cfg.totalParticles;
    if (cfg.interactionRadius !== undefined) this.interactionRadius = cfg.interactionRadius;
    if (cfg.maxSpeed !== undefined) this.maxSpeed = +cfg.maxSpeed;
    if (cfg.damping !== undefined) this.damping = +cfg.damping;
    if (cfg.seed !== undefined) this.seed = cfg.seed;
    if (cfg.trail !== undefined) this.trail = cfg.trail;
    if (cfg.glow !== undefined) this.glow = cfg.glow;
    if (cfg.glowSize !== undefined) this.glowSize = +cfg.glowSize;
    if (cfg.glowIntensity !== undefined) this.glowIntensity = +cfg.glowIntensity;
    if (cfg.bgColor !== undefined) this.bgColor = cfg.bgColor;
    if (cfg.showVectors !== undefined) this.showVectors = cfg.showVectors;
    if (cfg.showGrid !== undefined) this.showGrid = cfg.showGrid;
    if (cfg.life) {
      if (cfg.life.enabled !== undefined) this.lifeEnabled = cfg.life.enabled;
      if (cfg.life.energyDecay !== undefined) this.energyDecay = +cfg.life.energyDecay;
      if (cfg.life.collisionCost !== undefined) this.collisionCost = +cfg.life.collisionCost;
      if (cfg.life.feedRate !== undefined) this.feedRate = +cfg.life.feedRate;
      if (cfg.life.reproNeighbors !== undefined) this.reproNeighbors = +cfg.life.reproNeighbors;
      if (cfg.life.reproEnergy !== undefined) this.reproEnergy = +cfg.life.reproEnergy;
      if (cfg.life.maxParticles !== undefined) this.maxParticles = +cfg.life.maxParticles;
      if (cfg.life.reproCooldown !== undefined) this.reproCooldown = +cfg.life.reproCooldown;
    }
    if (cfg.types) this.types = cfg.types;
    if (cfg.matrix) this.matrix = cfg.matrix;
    if (cfg.soundEnabled !== undefined) {
      this.soundEnabled = cfg.soundEnabled;
      if (this.soundEnabled && globalThis.ui) {
        const soundBtn = document.getElementById('btn-sound');
        if (soundBtn) soundBtn.classList.add('active');
      }
    }

    const n = this.types.length;
    while (this.matrix.length < n) this.matrix.push(new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      while ((this.matrix[i] || []).length < n) this.matrix[i].push(0);
    }
    this.numTypes = n;

    this.buildGlowSprites();
    this.initParticles();
    ui.syncAll();
  }
}
