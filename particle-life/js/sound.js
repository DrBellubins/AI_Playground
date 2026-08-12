/**
 * SoundEngine — Web Audio API synthesizer for particle interactions.
 *
 * Behavior → sound mapping:
 *   swirl    → continuous sine wobble (tracks active interactions)
 *   clusters → white noise pluck
 *   galaxy   → square wave pluck
 *   random   → saw wave pluck
 *
 * No sound effect exceeds 500 ms.
 */
export class SoundEngine {
  /**
   * @param {Simulation} sim — reference to the simulation to read matrix/particles
   */
  constructor(sim) {
    this.sim = sim;

    // Create audio context lazily on first user gesture
    this.ctx = null;
    this.initialized = false;

    // Track active swirl interactions for continuous wobble
    this.swirlWobbles = new Map(); // key: "i,j" → { osc, gain, freq }

    // Cooldowns to avoid overlapping plucks
    this.cooldowns = {
      cluster: new Map(), // typePair → endTime
      galaxy: new Map(),
      random: new Map(),
    };

    // Debounce for swirl wobble — only trigger if interaction persists
    this.swirlDebounces = new Map(); // key: "i,j" → timeoutId
    this.swirlDebounceMs = 80;

    // Max simultaneous plucks per type
    this.maxPlucks = { cluster: 4, galaxy: 4, random: 4 };
    this.activePlucks = { cluster: 0, galaxy: 0, random: 0 };

    // Ensure cleanup on page unload
    window.addEventListener('beforeunload', () => this.dispose());
  }

  /* ---- Lazy init ---- */
  init() {
    if (this.initialized) return;
    try {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.initialized = true;
    } catch (e) {
      console.warn('Web Audio API not available:', e);
    }
  }

  /** Stop and release all audio nodes. */
  dispose() {
    if (this.ctx) {
      this.ctx.close().catch(() => {});
      this.ctx = null;
    }
    this.swirlWobbles.clear();
    for (const key of Object.keys(this.cooldowns)) {
      this.cooldowns[key].clear();
      this.activePlucks[key] = 0;
    }
  }

  /* ---- Public update called each frame ---- */
  update() {
    // Check if sound is enabled on the simulation
    if (!this.sim.soundEnabled) {
      // If previously enabled, clean up
      if (this.ctx && this.ctx.state !== 'closed') {
        this.dispose();
      }
      return;
    }

    if (!this.initialized || !this.ctx) return;

    const sim = this.sim;
    const particles = sim.particles;
    const matrix = sim.matrix;
    const grid = sim.grid;
    const radius = sim.interactionRadius;
    const numTypes = sim.numTypes;

    // Collect all active interactions this frame
    const activeInteractions = new Set(); // "i,j"
    const interactionTypes = new Map();   // "i,j" → { typeA, typeB, strength }

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const neighbors = grid.getNeighbors(p, radius);

      for (let j = 0; j < neighbors.length; j++) {
        const q = neighbors[j];
        if (q.type >= numTypes) continue;

        let dx = q.x - p.x;
        let dy = q.y - p.y;

        if (sim.wrap) {
          if (dx > sim.w * 0.5) dx -= sim.w;
          else if (dx < -sim.w * 0.5) dx += sim.w;
          if (dy > sim.h * 0.5) dy -= sim.h;
          else if (dy < -sim.h * 0.5) dy += sim.h;
        }

        const dSq = dx * dx + dy * dy;
        if (dSq >= radius * radius || dSq < 0.01) continue;

        const d = Math.sqrt(dSq);
        const strength = matrix[p.type]?.[q.type] ?? 0;
        if (strength === 0) continue;

        // Normalize key so (A,B) and (B,A) are the same
        const a = Math.min(p.type, q.type);
        const b = Math.max(p.type, q.type);
        const key = `${a},${b}`;
        activeInteractions.add(key);

        const dNorm = 1 - d / radius;
        const cur = interactionTypes.get(key);
        if (!cur || Math.abs(strength * dNorm) > Math.abs(cur.strength * dNorm)) {
          interactionTypes.set(key, { typeA: a, typeB: b, strength });
        }
      }
    }

    // --- Process swirl interactions → continuous sine wobble ---
    this._processSwirl(interactionTypes, activeInteractions);

    // --- Process non-swirl interactions → pluck sounds ---
    this._processPlucks(interactionTypes, activeInteractions);
  }

  /**
   * Swirl: continuous sine wobble whose pitch tracks interaction strength.
   * Debounced to avoid flicker.
   */
  _processSwirl(interactionTypes, activeInteractions) {
    // Start wobbles for newly active swirl interactions
    for (const [key, info] of interactionTypes) {
      const mat = this.sim.matrix;
      const sA = mat[info.typeA]?.[info.typeB] ?? 0;
      const sB = mat[info.typeB]?.[info.typeA] ?? 0;

      // Swirl preset has characteristic rotational forces
      if (!this._isSwirlLike(sA, sB)) continue;

      // Debounce: wait a few frames before starting
      if (!activeInteractions.has(key)) continue;

      if (!this.swirlWobbles.has(key)) {
        // Check debounce
        if (this.swirlDebounces.has(key)) continue;

        const timeout = setTimeout(() => {
          this.swirlDebounces.delete(key);
          this._startSwirlWobble(key, info);
        }, this.swirlDebounceMs);
        this.swirlDebounces.set(key, timeout);
      } else {
        // Update existing wobble frequency
        this._updateSwirlWobble(key, info);
      }
    }

    // Stop wobbles that are no longer active
    for (const [key] of this.swirlWobbles) {
      if (!activeInteractions.has(key)) {
        this._stopSwirlWobble(key);
      }
    }

    // Clean up debounce timers
    for (const [key, tid] of this.swirlDebounces) {
      if (activeInteractions.has(key)) continue;
      clearTimeout(tid);
      this.swirlDebounces.delete(key);
    }
  }

  /**
   * Determine if a pair of interaction strengths looks like swirl behavior.
   * Swirl has alternating positive/negative rotational forces.
   */
  _isSwirlLike(a, b) {
    // Swirl matrix has values like 1.5, -0.5, -1.5, 0.5
    // Check if both values are non-zero and at least one is in the swirl range
    if (Math.abs(a) < 0.1 && Math.abs(b) < 0.1) return false;
    // Swirl has characteristic alternating signs
    const swirlRange = (v) => Math.abs(v) >= 0.3 && Math.abs(v) <= 2.0;
    return swirlRange(a) || swirlRange(b);
  }

  _startSwirlWobble(key, info) {
    if (!this.ctx) return;
    const ctx = this.ctx;

    const osc = ctx.createOscillator();
    osc.type = 'sine';

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.06, ctx.currentTime + 0.05);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();

    this.swirlWobbles.set(key, { osc, gain, freq: 120 });
    this._updateSwirlWobble(key, info);
  }

  _updateSwirlWobble(key, info) {
    const entry = this.swirlWobbles.get(key);
    if (!entry) return;

    const strength = info.strength;
    // Map strength to frequency: stronger interaction = higher wobble
    const baseFreq = 80;
    const maxFreq = 400;
    const freq = baseFreq + Math.abs(strength) * (maxFreq - baseFreq);

    entry.osc.frequency.setTargetAtTime(freq, this.ctx.currentTime, 0.02);
  }

  _stopSwirlWobble(key) {
    const entry = this.swirlWobbles.get(key);
    if (!entry) return;

    const { osc, gain } = entry;
    if (this.ctx) {
      gain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.01);
    }
    setTimeout(() => {
      try { osc.stop(); } catch (_) {}
      osc.disconnect();
      gain.disconnect();
    }, 50);
    this.swirlWobbles.delete(key);
  }

  /**
   * Pluck sounds for non-swirl interactions.
   * Clusters → white noise, Galaxy → square, Random → saw
   */
  _processPlucks(interactionTypes, activeInteractions) {
    for (const [key, info] of interactionTypes) {
      if (!activeInteractions.has(key)) continue;

      const mat = this.sim.matrix;
      const sA = mat[info.typeA]?.[info.typeB] ?? 0;
      const sB = mat[info.typeB]?.[info.typeA] ?? 0;

      // Determine which behavior this pair belongs to
      let behavior = null;
      if (this._isClusterLike(sA, sB)) behavior = 'cluster';
      else if (this._isGalaxyLike(sA, sB)) behavior = 'galaxy';
      else if (this._isRandomLike(sA, sB)) behavior = 'random';

      if (!behavior) continue;

      // Check cooldown
      const cooldown = this.cooldowns[behavior];
      if (cooldown.has(key) && cooldown.get(key) > performance.now()) continue;

      // Check max plucks
      if (this.activePlucks[behavior] >= this.maxPlucks[behavior]) continue;

      // Trigger pluck
      this._triggerPluck(behavior, info);
      cooldown.set(key, performance.now() + 200); // 200ms cooldown between same pair
    }
  }

  /** Clusters: strong self-attraction + repulsion between types */
  _isClusterLike(a, b) {
    return Math.abs(a) > 0.5 && Math.abs(b) > 0.5 && a * b > 0;
  }

  /** Galaxy: radial bands of attraction/repulsion */
  _isGalaxyLike(a, b) {
    return Math.abs(a) > 0.3 && Math.abs(b) > 0.3;
  }

  /** Random: arbitrary values, generally smaller magnitude */
  _isRandomLike(a, b) {
    // If it's not cluster or galaxy-like, and has some force, treat as random
    return Math.abs(a) > 0.2 || Math.abs(b) > 0.2;
  }

  /** Trigger a pluck sound of the given behavior type */
  _triggerPluck(behavior, info) {
    if (!this.ctx) return;
    this.activePlucks[behavior]++;

    const ctx = this.ctx;
    const now = ctx.currentTime;
    const duration = Math.min(0.5, 0.15 + Math.abs(info.strength) * 0.1);

    if (behavior === 'cluster') {
      this._playWhiteNoisePluck(ctx, now, duration);
    } else if (behavior === 'galaxy') {
      this._playSquarePluck(ctx, now, duration, info);
    } else if (behavior === 'random') {
      this._playSawPluck(ctx, now, duration, info);
    }

    // Decrement counter after sound ends
    setTimeout(() => {
      this.activePlucks[behavior] = Math.max(0, this.activePlucks[behavior] - 1);
    }, duration * 1000);
  }

  /** White noise burst — used for cluster behavior */
  _playWhiteNoisePluck(ctx, now, duration) {
    const bufferSize = ctx.sampleRate * duration;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1);
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Low-pass filter for a softer pluck
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(3000, now);
    filter.frequency.exponentialRampToValueAtTime(300, now + duration);

    source.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    source.start(now);
    source.stop(now + duration);
  }

  /** Square wave pluck — used for galaxy behavior */
  _playSquarePluck(ctx, now, duration, info) {
    const osc = ctx.createOscillator();
    osc.type = 'square';

    // Pitch based on interaction strength
    const baseFreq = 150;
    const freq = baseFreq + Math.abs(info.strength) * 100;
    osc.frequency.setValueAtTime(freq, now);
    osc.frequency.exponentialRampToValueAtTime(Math.max(60, freq * 0.5), now + duration);

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.06, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Bandpass to tame the square wave harshness
    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(freq, now);
    filter.Q.setValueAtTime(2, now);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + duration);
  }

  /** Saw wave pluck — used for random behavior */
  _playSawPluck(ctx, now, duration, info) {
    const osc = ctx.createOscillator();
    osc.type = 'sawtooth';

    // Pitch based on interaction strength
    const baseFreq = 100;
    const freq = baseFreq + Math.abs(info.strength) * 120;
    osc.frequency.setValueAtTime(freq, now);
    osc.frequency.exponentialRampToValueAtTime(Math.max(50, freq * 0.4), now + duration);

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.05, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Low-pass to smooth the saw
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(2000, now);
    filter.frequency.exponentialRampToValueAtTime(200, now + duration);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + duration);
  }

  /**
   * Call this to initialize audio on user gesture (required by browsers).
   * Attach to a button click or similar.
   */
  enable() {
    this.init();
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  /** Pause all sound without destroying context. */
  pause() {
    if (this.ctx) {
      this.ctx.suspend();
    }
  }

  /** Resume sound after pause. */
  resume() {
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }
}
