import { mulberry32 } from './prng.js';
import { Particle } from './particle.js';
import { SpatialGrid } from './spatial-grid.js';
import { DEFAULT_TYPES, PRESETS } from './constants.js';

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
    this.seed = 42;
    this.trail = 30;
    this.bgColor = '#050508';
    this.showVectors = false;
    this.showGrid = false;
    this.wrap = false;

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

    // Fixed timestep
    this.fixedDt = 1 / 60;
    this.accTime = 0;
    this.lastTimestamp = 0;

    // FPS tracking
    this.frameCount = 0;
    this.fpsTime = 0;
    this.fps = 0;

    // World canvas — offscreen canvas at viewport size that holds
    // particles + trails in world coordinates. The main canvas draws
    // this canvas through the camera transform.
    this.worldCanvas = document.createElement('canvas');
    this.worldCtx = this.worldCanvas.getContext('2d');

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
        const p = new Particle(x, y, t);
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
        const p = new Particle(this.rng() * this.w, this.rng() * this.h, t);
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

      if (p.x < 0) { p.x = 0; p.vx *= -1; }
      else if (p.x >= this.w) { p.x = this.w - 1; p.vx *= -1; }
      if (p.y < 0) { p.y = 0; p.vy *= -1; }
      else if (p.y >= this.h) { p.y = this.h - 1; p.vy *= -1; }
    }
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
    ctx.drawImage(this.worldCanvas, 0, 0);
    ctx.restore();

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
  }

  /* ---- Main loop ---- */
  loop(timestamp) {
    requestAnimationFrame((t) => this.loop(t));

    // FPS
    this.frameCount++;
    if (timestamp - this.fpsTime >= 500) {
      this.fps = Math.round(this.frameCount / ((timestamp - this.fpsTime) / 1000));
      this.frameCount = 0;
      this.fpsTime = timestamp;
      if (globalThis.ui) globalThis.ui.updateStats();
    }

    if (!this.running) return;

    const delta = this.lastTimestamp ? (timestamp - this.lastTimestamp) / 1000 : this.fixedDt;
    this.lastTimestamp = timestamp;
    this.accTime += delta;

    let steps = 0;
    while (this.accTime >= this.fixedDt && steps < 5) {
      this.step();
      this.accTime -= this.fixedDt;
      steps++;
    }
    if (this.accTime > this.fixedDt * 5) this.accTime = 0;

    this.render();
  }

  start() {
    this.lastTimestamp = 0;
    this.accTime = 0;
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
      bgColor: this.bgColor,
      showVectors: this.showVectors,
      showGrid: this.showGrid,
      types: this.types.map(t => ({ ...t })),
      matrix: this.matrix.map(r => [...r]),
    };
  }

  importConfig(cfg) {
    if (cfg.totalParticles !== undefined) this.totalParticles = cfg.totalParticles;
    if (cfg.interactionRadius !== undefined) this.interactionRadius = cfg.interactionRadius;
    if (cfg.maxSpeed !== undefined) this.maxSpeed = +cfg.maxSpeed;
    if (cfg.damping !== undefined) this.damping = +cfg.damping;
    if (cfg.seed !== undefined) this.seed = cfg.seed;
    if (cfg.trail !== undefined) this.trail = cfg.trail;
    if (cfg.bgColor !== undefined) this.bgColor = cfg.bgColor;
    if (cfg.showVectors !== undefined) this.showVectors = cfg.showVectors;
    if (cfg.showGrid !== undefined) this.showGrid = cfg.showGrid;
    if (cfg.types) this.types = cfg.types;
    if (cfg.matrix) this.matrix = cfg.matrix;

    const n = this.types.length;
    while (this.matrix.length < n) this.matrix.push(new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      while ((this.matrix[i] || []).length < n) this.matrix[i].push(0);
    }
    this.numTypes = n;

    this.initParticles();
    ui.syncAll();
  }
}
