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
 *
 * Limits:
 *   - Max 32 simultaneous sounds globally
 *   - 2 second cooldown per particle based on zone entry events
 *     (not sound playback events)
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

    // --- Global sound limits ---
    this.maxActiveSounds = 32;
    this.activeSoundCount = 0;

    // --- Zone-entry cooldown ---
    // Tracks when particles last ENTERED an interaction zone (first appeared in neighbor list)
    this.zoneEntryCooldowns = new Map(); // particleIndex → endTime (ms)
    this.zoneEntryCooldownMs = 2000;

    // --- Swirl wobbles (continuous) ---
    this.swirlWobbles = new Map(); // key: "i,j" → { osc, gain }
    this.swirlDebounces = new Map(); // key: "i,j" → timeoutId
    this.swirlDebounceMs = 80;

    // --- Previous frame tracking for zone-entry detection ---
    this.prevInteractingParticles = new Set(); // particle indices that were interacting last frame

    // --- Cleanup on page unload ---
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
    this.swirlDebounces.clear();
    this.zoneEntryCooldowns.clear();
    this.prevInteractingParticles.clear();
    this.activeSoundCount = 0;
  }

  /** Check if a particle is on zone-entry cooldown. */
  _isOnZoneCooldown(particleIndex) {
    const endTime = this.zoneEntryCooldowns.get(particleIndex);
    if (endTime && endTime > performance.now()) {
      return true;
    }
    // Clean up expired entries
    if (endTime) {
      this.zoneEntryCooldowns.delete(particleIndex);
    }
    return false;
  }

  /** Set zone-entry cooldown for a particle. */
  _setZoneCooldown(particleIndex) {
    this.zoneEntryCooldowns.set(particleIndex, performance.now() + this.zoneEntryCooldownMs);
  }

  /** Try to claim a sound slot. Returns true if successful. */
  _claimSoundSlot() {
    if (this.activeSoundCount >= this.maxActiveSounds) {
      return false;
    }
    this.activeSoundCount++;
    return true;
  }

  /** Release a sound slot. */
  _releaseSoundSlot() {
    this.activeSoundCount = Math.max(0, this.activeSoundCount - 1);
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

    // --- Detect zone entries and exits ---
    const currentInteracting = new Set(); // particle indices currently interacting
    const newlyEntering = new Set();      // particles that just entered an interaction zone
    const interactionByParticle = new Map(); // particleIndex → [{ typeA, typeB, strength }]

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];

      // Zero-allocation neighbor scan (same visit order as before)
      grid.forEachNeighbor(i, p, radius, (qi) => {
        const q = particles[qi];
        if (q.type >= numTypes) return;

        let dx = q.x - p.x;
        let dy = q.y - p.y;

        if (sim.wrap) {
          if (dx > sim.w * 0.5) dx -= sim.w;
          else if (dx < -sim.w * 0.5) dx += sim.w;
          if (dy > sim.h * 0.5) dy -= sim.h;
          else if (dy < -sim.h * 0.5) dy += sim.h;
        }

        const dSq = dx * dx + dy * dy;
        if (dSq >= radius * radius || dSq < 0.01) return;

        const d = Math.sqrt(dSq);
        const strength = matrix[p.type]?.[q.type] ?? 0;
        if (strength === 0) return;

        // Normalize key so (A,B) and (B,A) are the same
        const a = Math.min(p.type, q.type);
        const b = Math.max(p.type, q.type);

        // Track interaction per particle
        const dNorm = 1 - d / radius;
        let list = interactionByParticle.get(i);
        if (!list) {
          list = [];
          interactionByParticle.set(i, list);
        }
        list.push({ typeA: a, typeB: b, strength: strength * dNorm });
      });
    }

    // Determine which particles are currently interacting
    for (const [particleIdx] of interactionByParticle) {
      currentInteracting.add(particleIdx);
    }

    // Detect zone entries: particles interacting now but not last frame
    for (const idx of currentInteracting) {
      if (!this.prevInteractingParticles.has(idx)) {
        newlyEntering.add(idx);
      }
    }

    // --- Process swirl interactions → continuous sine wobble ---
    this._processSwirl(interactionByParticle, currentInteracting);

    // --- Process non-swirl interactions → pluck sounds ---
    // Only trigger sounds for particles that just entered a zone AND are past cooldown
    this._processPlacks(interactionByParticle, newlyEntering);

    // Update previous frame tracking
    this.prevInteractingParticles = currentInteracting;
  }

  /**
   * Swirl: continuous sine wobble whose pitch tracks interaction strength.
   * Debounced to avoid flicker. Limited by global sound count.
   */
  _processSwirl(interactionByParticle, currentInteracting) {
    // Gather all swirl-like type-pairs from active interactions
    const swirlPairs = new Map(); // "i,j" → { typeA, typeB, strength }
    for (const [particleIdx, interactions] of interactionByParticle) {
      for (const info of interactions) {
        const mat = this.sim.matrix;
        const sA = mat[info.typeA]?.[info.typeB] ?? 0;
        const sB = mat[info.typeB]?.[info.typeA] ?? 0;

        if (!this._isSwirlLike(sA, sB)) continue;

        const key = `${info.typeA},${info.typeB}`;
        if (!swirlPairs.has(key)) {
          swirlPairs.set(key, info);
        } else {
          // Keep strongest interaction for this pair
          const cur = swirlPairs.get(key);
          if (Math.abs(info.strength) > Math.abs(cur.strength)) {
            swirlPairs.set(key, info);
          }
        }
      }
    }

    // Start wobbles for newly active swirl interactions
    for (const [key, info] of swirlPairs) {
      if (!currentInteracting.has(key)) continue;

      if (!this.swirlWobbles.has(key)) {
        // Check debounce
        if (this.swirlDebounces.has(key)) continue;

        // Check global sound limit
        if (!this._claimSoundSlot()) continue;

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
      if (!currentInteracting.has(key)) {
        this._stopSwirlWobble(key);
      }
    }

    // Clean up debounce timers
    for (const [key, tid] of this.swirlDebounces) {
      if (currentInteracting.has(key)) continue;
      clearTimeout(tid);
      this.swirlDebounces.delete(key);
    }
  }

  /**
   * Determine if a pair of interaction strengths looks like swirl behavior.
   */
  _isSwirlLike(a, b) {
    if (Math.abs(a) < 0.1 && Math.abs(b) < 0.1) return false;
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

    this.swirlWobbles.set(key, { osc, gain });
    this._updateSwirlWobble(key, info);
  }

  _updateSwirlWobble(key, info) {
    const entry = this.swirlWobbles.get(key);
    if (!entry) return;

    const strength = info.strength;
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
    this._releaseSoundSlot();
  }

  /**
   * Pluck sounds for non-swirl interactions.
   * Clusters → white noise, Galaxy → square, Random → saw
   * Only triggered when particles ENTER an interaction zone, not while staying.
   */
  _processPlacks(interactionByParticle, newlyEntering) {
    for (const particleIdx of newlyEntering) {
      const interactions = interactionByParticle.get(particleIdx);
      if (!interactions) continue;

      // Check zone-entry cooldown
      if (this._isOnZoneCooldown(particleIdx)) continue;

      // Find the strongest non-swirl interaction for this particle
      let bestInteraction = null;
      let bestBehavior = null;

      for (const info of interactions) {
        const mat = this.sim.matrix;
        const sA = mat[info.typeA]?.[info.typeB] ?? 0;
        const sB = mat[info.typeB]?.[info.typeA] ?? 0;

        let behavior = null;
        if (this._isClusterLike(sA, sB)) behavior = 'cluster';
        else if (this._isGalaxyLike(sA, sB)) behavior = 'galaxy';
        else if (this._isRandomLike(sA, sB)) behavior = 'random';

        if (!behavior) continue;

        // Pick the strongest interaction
        if (!bestInteraction || Math.abs(info.strength) > Math.abs(bestInteraction.strength)) {
          bestInteraction = info;
          bestBehavior = behavior;
        }
      }

      if (!bestInteraction || !bestBehavior) continue;

      // Check global sound limit
      if (!this._claimSoundSlot()) continue;

      // Set zone-entry cooldown and trigger pluck
      this._setZoneCooldown(particleIdx);
      this._triggerPluck(bestBehavior, bestInteraction);
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
    return Math.abs(a) > 0.2 || Math.abs(b) > 0.2;
  }

  /** Trigger a pluck sound of the given behavior type */
  _triggerPluck(behavior, info) {
    if (!this.ctx) return;
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

    // Release sound slot after sound ends
    setTimeout(() => {
      this._releaseSoundSlot();
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
