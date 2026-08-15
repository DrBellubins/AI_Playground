// Headless harness: stubs the minimal browser APIs so the Simulation module
// can be instantiated and driven in Node. Exercises BOTH the CPU fallback and
// the GPU code paths (with a fake engine).
import assert from 'node:assert';

// ---------- 2D context stub ----------
function make2d() {
  const noop = () => {};
  const grad = { addColorStop: noop };
  return {
    canvas: null,
    setTransform: noop, clearRect: noop, fillRect: noop,
    save: noop, restore: noop, translate: noop, scale: noop,
    beginPath: noop, moveTo: noop, lineTo: noop, stroke: noop, fill: noop,
    arc: noop, drawImage: noop, setLineDash: noop,
    createLinearGradient: () => grad, createRadialGradient: () => grad,
    fillStyle: '', strokeStyle: '', lineWidth: 1, lineCap: 'round',
    globalAlpha: 1, globalCompositeOperation: 'source-over',
    shadowBlur: 0, shadowColor: '', font: '', textBaseline: '', textAlign: '',
  };
}

// ---------- canvas stub ----------
function makeCanvas(w = 1280, h = 720, dpr = 1) {
  let ctx2d = null, gpuCtx = null;
  const canvas = {
    width: w, height: h,
    clientWidth: w, clientHeight: h,
    parentElement: { clientWidth: w, clientHeight: h },
    _type: null,
    getContext(type) {
      if (type === '2d') {
        if (gpuCtx) throw new Error('canvas already has a GPU context');
        ctx2d = make2d(); return ctx2d;
      }
      if (type === 'webgpu') {
        if (ctx2d) throw new Error('canvas already has a 2d context');
        gpuCtx = true; return {};
      }
      return null;
    },
    addEventListener: () => {}, removeEventListener: () => {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: w, height: h }),
    requestPointerLock: () => {},
  };
  return canvas;
}

// ---------- AudioContext stub ----------
function makeAudioNode() {
  return { connect: () => ({}), disconnect: noop, start: noop, stop: noop,
    frequency: { value: 440, setTargetAtTime: noop, linearRampToValueAtTime: noop, setValueAtTime: noop },
    gain: { value: 0, setTargetAtTime: noop, linearRampToValueAtTime: noop, setValueAtTime: noop },
    pan: { value: 0, setTargetAtTime: noop },
    Q: { value: 1 }, type: 'lowpass',
    threshold: { value: -24 }, ratio: { value: 4 }, knee: { value: 12 },
  };
}
const noop = () => {};
class AudioContext {
  constructor() { this.destination = { connect: noop }; this.currentTime = 0; this.state = 'running'; }
  createOscillator() { return makeAudioNode(); }
  createGain() { return makeAudioNode(); }
  createDynamicsCompressor() { return makeAudioNode(); }
  createBiquadFilter() { return makeAudioNode(); }
  createStereoPanner() { return makeAudioNode(); }
  createBufferSource() { return makeAudioNode(); }
  resume() { return Promise.resolve(); }
}

// ---------- globals ----------
globalThis.window = {
  devicePixelRatio: 1,
  addEventListener: noop, removeEventListener: noop,
  innerWidth: 1280, innerHeight: 720,
};
globalThis.document = {
  createElement: (tag) => makeCanvas(),
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 1280, height: 720 }),
};
globalThis.navigator = { gpu: undefined, userAgent: 'node' }; // no WebGPU → CPU path
globalThis.AudioContext = AudioContext;
globalThis.performance = { now: () => Date.now() };
let rafQ = [];
globalThis.requestAnimationFrame = (cb) => { rafQ.push(cb); return rafQ.length; };
globalThis.cancelAnimationFrame = noop;

const tick = () => new Promise((r) => setTimeout(r, 0));
async function pumpRaf(n) { for (let i = 0; i < n; i++) { const q = rafQ; rafQ = []; const t = Date.now(); q.forEach((cb) => cb(t)); await tick(); } }

const { Simulation } = await import('../js/simulation.js');

// =====================================================================
console.log('== TEST 1: CPU fallback path ==');
{
  const canvas = makeCanvas();
  const sim = new Simulation(canvas);
  await tick(); // let _initContext settle (CPU → synchronous)
  assert.ok(sim.ctx, 'expected a 2d context (CPU fallback)');
  assert.ok(!sim.gpu, 'expected no GPU in CPU mode');
  assert.ok(sim._contextReady, 'context should be ready');
  assert.strictEqual(sim.particles.length, 1200, 'initial particle count');

  sim.start();
  await pumpRaf(10);
  assert.ok(sim.particles.length > 0, 'particles alive after frames');
  const p = sim.particles[0];
  assert.ok(typeof p.x === 'number' && !Number.isNaN(p.x), 'particle x is a number');
  // grid should be built
  assert.ok(sim.grid.cols >= 1, 'grid built');
  console.log('   CPU: particles=', sim.particles.length,
              'x=', p.x.toFixed(2), 'y=', p.y.toFixed(2),
              'v=', Math.hypot(p.vx, p.vy).toFixed(3), 'OK');
}

// =====================================================================
console.log('== TEST 2: GPU path (fake engine) ==');
{
  // A faithful-ish fake engine: ping-pong data, kills one, births one.
  class FakeGpu {
    constructor() {
      this.maxParticles = 64;
      this._data = new Float32Array(this.maxParticles * 12);
      this.computeCalls = 0; this.renderCalls = 0; this.paramsSet = 0;
      for (let i = 0; i < this.maxParticles; i++) this._data[i * 12 + 11] = i; // id = slot
    }
    static isSupported() { return true; }
    async init() { return true; }
    setTypes() {} setMatrix() {} setBgColor() {} setGlowParams() {} setCamera() {}
    setParams() { this.paramsSet++; }
    uploadParticles(list) {
      this._data.fill(0);
      for (let i = 0; i < this.maxParticles; i++) this._data[i * 12 + 11] = i;
      for (let i = 0; i < list.length; i++) {
        const o = i * 12;
        this._data[o] = list[i].x; this._data[o + 1] = list[i].y;
        this._data[o + 6] = list[i].type; this._data[o + 10] = 1; this._data[o + 11] = list[i]._index;
      }
    }
    submitCompute() {
      this.computeCalls++;
      // kill slot 0 (id 0), birth: slot 64-1? keep in bounds → slot 5 alive id 99
      this._data[0 * 12 + 10] = 0;               // kill id 0
      const o = 5 * 12;                           // spawn at slot 5
      this._data[o] = 200; this._data[o + 1] = 200; this._data[o + 10] = 1; this._data[o + 11] = 99;
    }
    submitRender() { this.renderCalls++; }
    async readParticles() { return new Float32Array(this._data); }
    resize() {}
  }

  const canvas = makeCanvas();
  const sim = new Simulation(canvas);
  // Force the GPU path: install fake engine, skip async init.
  sim.gpu = new FakeGpu();
  sim.gpu.uploadParticles(sim.particles);
  sim._resetDeathTracking();
  sim._contextReady = true;
  sim.ctx = null;

  assert.strictEqual(sim._prevAlive.size, 1200, 'prev alive set initialized');

  // Drive one frame's worth of GPU logic in order.
  sim._gpuSetParams();
  assert.ok(sim.gpu.paramsSet >= 1, 'params pushed');
  sim.step();                       // → submitCompute
  assert.ok(sim.gpu.computeCalls >= 1, 'compute submitted');
  sim._gpuRender();                 // → submitRender
  assert.ok(sim.gpu.renderCalls >= 1, 'render submitted');
  sim._kickoffReadback();
  await tick();
  sim._processPendingReadback();

  // slot0 (id0) killed, slot5 (id99) born. alive ids = {1,2,3,4,...,1199,99}
  assert.strictEqual(sim.particles.length, 1200, 'population stays 1200 (1 death + 1 birth)');
  assert.strictEqual(sim.deaths, 1, 'one death detected');
  assert.strictEqual(sim.births, 1, 'one birth detected');
  assert.ok(sim.deathFx.length > 0, 'death FX spawned from readback');
  assert.ok(!sim._prevAlive.has(0), 'id 0 no longer alive');
  assert.ok(sim._prevAlive.has(99), 'id 99 alive');
  const id99 = sim.particles.find((p) => p._index === 99);
  assert.ok(id99 && id99.x === 200 && id99.y === 200, 'id99 mirror has readback position');
  console.log('   GPU: particles=', sim.particles.length,
              'deaths=', sim.deaths, 'births=', sim.births,
              'deathFx=', sim.deathFx.length, 'compute=', sim.gpu.computeCalls, 'render=', sim.gpu.renderCalls, 'OK');

  // Run a couple more frames to make sure it stays stable.
  for (let f = 0; f < 3; f++) {
    sim._gpuSetParams(); sim.step(); sim._gpuRender(); sim._kickoffReadback();
    await tick(); sim._processPendingReadback();
  }
  assert.ok(sim.particles.length > 0, 'particles still alive');
  console.log('   GPU multi-frame stable, particles=', sim.particles.length, 'OK');
}

// =====================================================================
console.log('\nALL TESTS PASSED');
