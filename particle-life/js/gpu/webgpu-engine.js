/**
 * WebGPUEngine — GPU compute + render engine for the particle life simulation.
 *
 * Handles:
 *   - Spatial binning (count → prefix sum → sort)
 *   - Force computation + integration + life cycle (compute shaders)
 *   - Trail accumulation (ping-pong textures)
 *   - Particle rendering (circles, glow, death FX, vectors, grid)
 *   - Particle data readback for CPU-side logic (sound, death FX spawning)
 */
import {
    BIN_CLEAR, BIN_COUNT, PREFIX_SUM, BIN_SORT, COMPUTE_ALL, SPAWN_RESOLVE,
    TRAIL_FADE, CIRCLE, CIRCLE_TRAIL, GLOW, COMPOSITE, LINE_SIMPLE, DEATH_FX,
} from './shaders.js';

const PARTICLE_FLOATS = 12; // x,y,prevX,prevY,vx,vy,type,energy,age,reproCooldown,alive,id
const PARTICLE_BYTES = PARTICLE_FLOATS * 4; // 48
const MAX_PARTICLES = 8000;
const MAX_BINS = 6000; // max grid cells (96×54 = 5184, with headroom)
const MAX_TYPES = 16;
const MAX_FX = 400;
const TYPE_INFO_FLOATS = 4; // r,g,b,size per type

// Particle field indices
const P_X = 0, P_Y = 1, P_PX = 2, P_PY = 3, P_VX = 4, P_VY = 5,
      P_TYPE = 6, P_ENERGY = 7, P_AGE = 8, P_COOLDOWN = 9, P_ALIVE = 10, P_ID = 11;

export class WebGPUEngine {
    constructor() {
        this.device = null;
        this.queue = null;
        this.canvasCtx = null;
        this.canvasFormat = null;
        this.supported = false;

        // State
        this.maxParticles = MAX_PARTICLES;
        this.gridW = 0;
        this.gridH = 0;
        this.currentBuf = 'A'; // which particle buffer holds current state
        this.trailCurrent = 'A'; // which trail texture holds current trails
        this.aliveCount = 0;

        // Simulation params (cached for uniform updates)
        this._params = null;
        this._camera = null;
        this._bgColor = null;
        this._glowParams = null;

        // Readback (ping-pong — a buffer can't be a copy destination while a
        // previous frame's mapAsync is still mapped/mapping)
        this._readbackBufA = null;
        this._readbackBufB = null;
        this._rbFlip = false;

        // CPU-side death FX
        this.deathFX = [];

        // Buffers, pipelines, bind groups — created in init()
    }

    /** Check if WebGPU is available. */
    static isSupported() {
        return !!navigator.gpu;
    }

    /** Initialize the WebGPU engine. */
    async init(canvas) {
        if (!WebGPUEngine.isSupported()) return false;

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return false;
            this.device = await adapter.requestDevice();
            this.queue = this.device.queue;
            this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();

            this.canvasCtx = canvas.getContext('webgpu');
            this.canvasCtx.configure({
                device: this.device,
                format: this.canvasFormat,
                alphaMode: 'opaque',
            });

            this._createBuffers();
            this._createTextures(canvas);
            this._createPipelines();
            this.supported = true;
            return true;
        } catch (e) {
            console.warn('WebGPU init failed:', e);
            this.supported = false;
            return false;
        }
    }

    /* ---- Buffer creation ---- */
    _createBuffers() {
        const d = this.device;
        const q = this.queue;

        // Particle buffers (ping-pong + sorted)
        const pSize = MAX_PARTICLES * PARTICLE_BYTES;
        this.particleBufA = d.createBuffer({ size: pSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        this.particleBufB = d.createBuffer({ size: pSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        this.sortedBuf = d.createBuffer({ size: pSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });

        // Bin data (ping-pong for prefix sum)
        const binSize = (MAX_BINS + 1) * 4;
        this.binDataA = d.createBuffer({ size: binSize, usage: GPUBufferUsage.STORAGE });
        this.binDataB = d.createBuffer({ size: binSize, usage: GPUBufferUsage.STORAGE });

        // Sort counters
        this.sortCountersBuf = d.createBuffer({ size: MAX_BINS * 4, usage: GPUBufferUsage.STORAGE });

        // Free list
        this.freeListBuf = d.createBuffer({ size: MAX_PARTICLES * 4, usage: GPUBufferUsage.STORAGE });
        this.freeCountBuf = d.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
        this.freeHeadBuf = d.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
        // Spawn requests (SpawnReq = 10 floats each), one per slot, resolved in a
        // separate dispatch to avoid a write race between parent and free slot.
        this.spawnReqBuf = d.createBuffer({ size: MAX_PARTICLES * 10 * 4, usage: GPUBufferUsage.STORAGE });

        // Uniforms
        this.paramsBuf = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.cameraBuf = d.createBuffer({ size: 36, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // 9 floats (Camera struct)
        this.typeInfoBuf = d.createBuffer({ size: MAX_TYPES * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.matrixBuf = d.createBuffer({ size: MAX_TYPES * MAX_TYPES * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.psParamsBuf = d.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.fadeFactorBuf = d.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.bgColorBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.glowParamsBuf = d.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.glowSizeBuf = d.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

        // Death FX
        this.fxBuf = d.createBuffer({ size: MAX_FX * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

        // Readback (ping-pong pair)
        this._readbackBufA = d.createBuffer({ size: pSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this._readbackBufB = d.createBuffer({ size: pSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this._rbFlip = false;

        // Line vertex buffer (for vectors & grid) — max 10000 lines × 6 verts × 24 bytes
        this.lineBuf = d.createBuffer({ size: 10000 * 6 * 24, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        this._lineVertexCount = 0;

        // Initialize type info with defaults
        this._typeInfoData = new Float32Array(MAX_TYPES * 4);
        this._matrixData = new Float32Array(MAX_TYPES * MAX_TYPES);
    }

    /* ---- Texture creation ---- */
    _createTextures(canvas) {
        const d = this.device;
        const dpr = window.devicePixelRatio || 1;
        const w = Math.floor(canvas.clientWidth * dpr) || 800;
        const h = Math.floor(canvas.clientHeight * dpr) || 600;

        // Use logical resolution for trail (saves memory, visually identical for fading trails)
        this.trailW = Math.max(1, canvas.clientWidth);
        this.trailH = Math.max(1, canvas.clientHeight);

        const trailDesc = {
            size: [this.trailW, this.trailH],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        };

        this.trailTexA = d.createTexture(trailDesc);
        this.trailTexB = d.createTexture(trailDesc);
        this.trailViewA = this.trailTexA.createView();
        this.trailViewB = this.trailTexB.createView();

        this.trailSampler = d.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });

        this.canvasW = w;
        this.canvasH = h;
    }

    /** Recreate trail textures on resize. */
    resize(canvas) {
        const dpr = window.devicePixelRatio || 1;
        this.canvasW = Math.floor(canvas.clientWidth * dpr) || 800;
        this.canvasH = Math.floor(canvas.clientHeight * dpr) || 600;
        this.trailW = Math.max(1, canvas.clientWidth);
        this.trailH = Math.max(1, canvas.clientHeight);

        if (this.trailTexA) {
            this.trailTexA.destroy();
            this.trailTexB.destroy();
        }

        const trailDesc = {
            size: [this.trailW, this.trailH],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        };

        this.trailTexA = this.device.createTexture(trailDesc);
        this.trailTexB = this.device.createTexture(trailDesc);
        this.trailViewA = this.trailTexA.createView();
        this.trailViewB = this.trailTexB.createView();

        // Reconfigure canvas
        this.canvasCtx.configure({
            device: this.device,
            format: this.canvasFormat,
            alphaMode: 'opaque',
        });
    }

    /* ---- Pipeline creation ---- */
    _createPipelines() {
        const d = this.device;

        // --- Compute pipelines ---
        this.binClearPipeline = d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code: BIN_CLEAR }), entryPoint: 'main' },
        });

        this.binCountPipeline = d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code: BIN_COUNT }), entryPoint: 'main' },
        });

        this.prefixSumPipeline = d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code: PREFIX_SUM }), entryPoint: 'main' },
        });

        this.binSortPipeline = d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code: BIN_SORT }), entryPoint: 'main' },
        });

        this.computeAllPipeline = d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code: COMPUTE_ALL }), entryPoint: 'main' },
        });

        this.spawnResolvePipeline = d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code: SPAWN_RESOLVE }), entryPoint: 'main' },
        });

        // --- Render pipelines ---
        const noBlend = { color: { format: this.canvasFormat, alphaMode: 'opaque' } };

        // Trail fade
        this.trailFadePipeline = d.createRenderPipeline({
            layout: 'auto',
            vertex: { module: d.createShaderModule({ code: TRAIL_FADE }), entryPoint: 'vs' },
            fragment: { module: d.createShaderModule({ code: TRAIL_FADE }), entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
            primitive: { topology: 'triangle-list' },
        });

        // Circle (used for both trail dots and screen particles)
        const circleModule = d.createShaderModule({ code: CIRCLE });
        this.circlePipeline = d.createRenderPipeline({
            layout: 'auto',
            vertex: { module: circleModule, entryPoint: 'vs' },
            fragment: { module: circleModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm', blend: { color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] },
            primitive: { topology: 'triangle-list' },
        });

        // Circle for trail (additive blending onto the world-locked trail texture)
        const circleTrailModule = d.createShaderModule({ code: CIRCLE_TRAIL });
        this.circleTrailPipeline = d.createRenderPipeline({
            layout: 'auto',
            vertex: { module: circleTrailModule, entryPoint: 'vs' },
            fragment: { module: circleTrailModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm', blend: { color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' }, alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' } } }] },
            primitive: { topology: 'triangle-list' },
        });

        // Glow (additive)
        const glowModule = d.createShaderModule({ code: GLOW });
        this.glowPipeline = d.createRenderPipeline({
            layout: 'auto',
            vertex: { module: glowModule, entryPoint: 'vs' },
            fragment: { module: glowModule, entryPoint: 'fs', targets: [{ format: this.canvasFormat, blend: { color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' }, alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' } } }] },
            primitive: { topology: 'triangle-list' },
        });

        // Composite (background + trail)
        this.compositePipeline = d.createRenderPipeline({
            layout: 'auto',
            vertex: { module: d.createShaderModule({ code: COMPOSITE }), entryPoint: 'vs' },
            fragment: { module: d.createShaderModule({ code: COMPOSITE }), entryPoint: 'fs', targets: [{ format: this.canvasFormat }] },
            primitive: { topology: 'triangle-list' },
        });

        // Line (vectors & grid)
        const lineModule = d.createShaderModule({ code: LINE_SIMPLE });
        this.linePipeline = d.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: lineModule, entryPoint: 'vs',
                buffers: [{ arrayStride: 24, attributes: [
                    { shaderLocation: 0, offset: 0, format: 'float32x2' },
                    { shaderLocation: 1, offset: 8, format: 'float32x4' },
                ]}],
            },
            fragment: { module: lineModule, entryPoint: 'fs', targets: [{ format: this.canvasFormat, blend: { color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] },
            primitive: { topology: 'triangle-list' },
        });

        // Death FX (additive)
        const fxModule = d.createShaderModule({ code: DEATH_FX });
        this.fxPipeline = d.createRenderPipeline({
            layout: 'auto',
            vertex: { module: fxModule, entryPoint: 'vs' },
            fragment: { module: fxModule, entryPoint: 'fs', targets: [{ format: this.canvasFormat, blend: { color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' }, alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' } } }] },
            primitive: { topology: 'triangle-list' },
        });
    }

    /* ---- Parameter updates ---- */

    /**
     * Set simulation parameters. Call before each compute step.
     * @param {object} p - { worldW, worldH, radius, maxSpeed, damping, dt, wrap, lifeEnabled,
     *                       energyDecay, collisionCost, feedRate, reproNeighbors, reproEnergy,
     *                       reproCooldown, numTypes, trail }
     */
    setParams(p) {
        this._params = p;

        // Compute grid dimensions, growing the cell size if needed so the grid
        // never exceeds MAX_BINS cells (the bin buffers are fixed at MAX_BINS;
        // an overflow here is an out-of-bounds GPU write → device lost). The
        // 3x3 neighbor search stays correct as long as cellSize >= radius.
        let cellSize = p.radius;
        let gridW = Math.max(1, Math.ceil(p.worldW / cellSize));
        let gridH = Math.max(1, Math.ceil(p.worldH / cellSize));
        while (gridW * gridH > MAX_BINS) {
            cellSize *= 1.15;
            gridW = Math.max(1, Math.ceil(p.worldW / cellSize));
            gridH = Math.max(1, Math.ceil(p.worldH / cellSize));
        }
        this.gridW = gridW;
        this.gridH = gridH;

        const data = new Float32Array(24);
        data[0] = p.worldW;
        data[1] = p.worldH;
        data[2] = p.radius;
        data[3] = p.maxSpeed;
        data[4] = p.damping;
        data[5] = p.dt;
        data[6] = p.wrap ? 1.0 : 0.0;
        data[7] = p.lifeEnabled ? 1.0 : 0.0;
        data[8] = p.energyDecay;
        data[9] = p.collisionCost;
        data[10] = p.feedRate;
        data[11] = p.reproNeighbors;
        data[12] = p.reproEnergy;
        data[13] = p.reproCooldown;
        data[14] = p.numTypes;
        data[15] = gridW;
        data[16] = gridH;
        data[17] = cellSize; // cellSize = radius (clamped to fit the bin buffers)
        data[18] = this.maxParticles;
        data[19] = p.trail || 0;
        this.queue.writeBuffer(this.paramsBuf, 0, data);
    }

    /** Set camera parameters. */
    setCamera(viewX, viewY, zoom) {
        this.queue.writeBuffer(this.cameraBuf, 0, new Float32Array([
            viewX, viewY, zoom,
            this._params ? this._params.worldW : 800,
            this._params ? this._params.worldH : 600,
            this.canvasW, this.canvasH,
            0, 0,
        ]));
    }

    /** Set per-type color and size data. */
    setTypes(types) {
        const data = this._typeInfoData;
        data.fill(0);
        for (let i = 0; i < types.length && i < MAX_TYPES; i++) {
            const t = types[i];
            const hex = (t.color || '#ffffff').replace('#', '');
            const r = parseInt(hex.substring(0, 2), 16) / 255;
            const g = parseInt(hex.substring(2, 4), 16) / 255;
            const b = parseInt(hex.substring(4, 6), 16) / 255;
            data[i * 4] = r;
            data[i * 4 + 1] = g;
            data[i * 4 + 2] = b;
            data[i * 4 + 3] = t.size || 2;
        }
        this.queue.writeBuffer(this.typeInfoBuf, 0, data);
    }

    /** Set interaction matrix (flat row-major, numTypes × numTypes). */
    setMatrix(matrix, numTypes) {
        const data = this._matrixData;
        data.fill(0);
        for (let i = 0; i < numTypes && i < MAX_TYPES; i++) {
            for (let j = 0; j < numTypes && j < MAX_TYPES; j++) {
                data[i * numTypes + j] = matrix[i]?.[j] ?? 0;
            }
        }
        this.queue.writeBuffer(this.matrixBuf, 0, data);
    }

    /** Upload particle data from CPU. */
    uploadParticles(particles) {
        const data = new Float32Array(this.maxParticles * PARTICLE_FLOATS);
        for (let i = 0; i < particles.length && i < this.maxParticles; i++) {
            const p = particles[i];
            const o = i * PARTICLE_FLOATS;
            data[o + P_X] = p.x;
            data[o + P_Y] = p.y;
            data[o + P_PX] = p.prevX;
            data[o + P_PY] = p.prevY;
            data[o + P_VX] = p.vx;
            data[o + P_VY] = p.vy;
            data[o + P_TYPE] = p.type;
            data[o + P_ENERGY] = p.energy !== undefined ? p.energy : 1;
            data[o + P_AGE] = p.age || 0;
            data[o + P_COOLDOWN] = p.reproCooldown || 0;
            data[o + P_ALIVE] = 1;
            data[o + P_ID] = i;
        }
        // Mark unused slots as dead
        for (let i = particles.length; i < this.maxParticles; i++) {
            const o = i * PARTICLE_FLOATS;
            data[o + P_ALIVE] = 0;
            data[o + P_ID] = i;
        }

        // Write to the current buffer
        const buf = this.currentBuf === 'A' ? this.particleBufA : this.particleBufB;
        this.queue.writeBuffer(buf, 0, data);
        this.aliveCount = particles.length;
    }

    /** Set background color. */
    setBgColor(r, g, b) {
        this.queue.writeBuffer(this.bgColorBuf, 0, new Float32Array([r, g, b, 1]));
    }

    /** Set glow parameters. */
    setGlowParams(glowSize, glowIntensity) {
        this.queue.writeBuffer(this.glowParamsBuf, 0, new Float32Array([glowSize, glowIntensity]));
        this.queue.writeBuffer(this.glowSizeBuf, 0, new Float32Array([glowSize]));
    }

    /* ---- Compute step ---- */

    /**
     * Submit one physics step to the GPU queue.
     * Must be called after setParams().
     */
    submitCompute() {
        const d = this.device;
        const q = this.queue;
        const encoder = d.createCommandEncoder();

        const curBuf = this.currentBuf === 'A' ? this.particleBufA : this.particleBufB;
        const outBuf = this.currentBuf === 'A' ? this.particleBufB : this.particleBufA;
        const numBins = this.gridW * this.gridH;
        const numParticles = this.maxParticles;
        const wgCount = Math.ceil(numParticles / 64);
        const binWgCount = Math.ceil((numBins + 1) / 256);

        // 1. Clear bins, sort counters, free list
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.binClearPipeline);
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.binClearPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.binDataA } },
                    { binding: 1, resource: { buffer: this.sortCountersBuf } },
                    { binding: 2, resource: { buffer: this.freeCountBuf } },
                    { binding: 3, resource: { buffer: this.freeHeadBuf } },
                    { binding: 4, resource: { buffer: this.paramsBuf } },
                ],
            }));
            pass.dispatchWorkgroups(binWgCount);
            pass.end();
        }

        // 2. Count particles per bin + build free list
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.binCountPipeline);
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.binCountPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: curBuf } },
                    { binding: 1, resource: { buffer: this.binDataA } },
                    { binding: 2, resource: { buffer: this.freeListBuf } },
                    { binding: 3, resource: { buffer: this.freeCountBuf } },
                    { binding: 4, resource: { buffer: this.paramsBuf } },
                ],
            }));
            pass.dispatchWorkgroups(wgCount);
            pass.end();
        }

        // 3. Parallel prefix sum (ping-pong)
        {
            const count = numBins + 1;
            const steps = Math.ceil(Math.log2(count));
            const evenSteps = Math.ceil(steps / 2) * 2; // always even

            for (let s = 0; s < evenSteps; s++) {
                const stepSize = 1 << s;
                q.writeBuffer(this.psParamsBuf, 0, new Uint32Array([stepSize, count]));

                const input = (s % 2 === 0) ? this.binDataA : this.binDataB;
                const output = (s % 2 === 0) ? this.binDataB : this.binDataA;

                const pass = encoder.beginComputePass();
                pass.setPipeline(this.prefixSumPipeline);
                pass.setBindGroup(0, d.createBindGroup({
                    layout: this.prefixSumPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: input } },
                        { binding: 1, resource: { buffer: output } },
                        { binding: 2, resource: { buffer: this.psParamsBuf } },
                    ],
                }));
                pass.dispatchWorkgroups(Math.ceil(count / 64));
                pass.end();
            }

            // After evenSteps (even number), result is in binDataA
        }

        // 4. Sort particles into bins
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.binSortPipeline);
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.binSortPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: curBuf } },
                    { binding: 1, resource: { buffer: this.sortedBuf } },
                    { binding: 2, resource: { buffer: this.binDataA } }, // offsets
                    { binding: 3, resource: { buffer: this.sortCountersBuf } },
                    { binding: 4, resource: { buffer: this.paramsBuf } },
                ],
            }));
            pass.dispatchWorkgroups(wgCount);
            pass.end();
        }

        // 5. Compute forces + integration + life cycle
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.computeAllPipeline);
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.computeAllPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: curBuf } },
                    { binding: 1, resource: { buffer: outBuf } },
                    { binding: 2, resource: { buffer: this.sortedBuf } },
                    { binding: 3, resource: { buffer: this.binDataA } }, // offsets
                    { binding: 4, resource: { buffer: this.paramsBuf } },
                    { binding: 5, resource: { buffer: this.matrixBuf } },
                    { binding: 6, resource: { buffer: this.freeListBuf } },
                    { binding: 7, resource: { buffer: this.freeCountBuf } },
                    { binding: 8, resource: { buffer: this.freeHeadBuf } },
                    { binding: 9, resource: { buffer: this.spawnReqBuf } },
                ],
            }));
            pass.dispatchWorkgroups(wgCount);
            pass.end();
        }

        // 5b. Resolve spawn requests into free slots (race-free).
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.spawnResolvePipeline);
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.spawnResolvePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.spawnReqBuf } },
                    { binding: 1, resource: { buffer: outBuf } },
                    { binding: 2, resource: { buffer: this.paramsBuf } },
                ],
            }));
            pass.dispatchWorkgroups(wgCount);
            pass.end();
        }

        // 6. Copy to readback buffer (ping-pong; read back in readParticles)
        encoder.copyBufferToBuffer(outBuf, 0, this._rbCurrent, 0, numParticles * PARTICLE_BYTES);

        q.submit([encoder.finish()]);

        // Swap ping-pong
        this.currentBuf = this.currentBuf === 'A' ? 'B' : 'A';
    }

    /* ---- Render ---- */

    /**
     * Submit render passes to the GPU queue.
     * @param {object} opts - { camera: {viewX, viewY, zoom}, glow: bool, glowSize, glowIntensity,
     *                          trail: number, showVectors: bool, showGrid: bool, deathFX: array }
     */
    submitRender(opts) {
        const d = this.device;
        const q = this.queue;
        const encoder = d.createCommandEncoder();
        const canvas = this.canvasCtx.getCurrentTexture();
        const view = canvas.createView();

        const curBuf = this.currentBuf === 'A' ? this.particleBufA : this.particleBufB;
        const numParticles = this.maxParticles;
        const trailOn = opts.trail > 0;
        const trailCurrentView = this.trailCurrent === 'A' ? this.trailViewA : this.trailViewB;
        const trailOtherView = this.trailCurrent === 'A' ? this.trailViewB : this.trailViewA;

        // 1. Trail fade (if trails enabled)
        if (trailOn) {
            const fadeFactor = Math.min(0.98, opts.trail / 32);
            q.writeBuffer(this.fadeFactorBuf, 0, new Float32Array([fadeFactor]));

            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: trailOtherView,
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.trailFadePipeline);
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.trailFadePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: trailCurrentView },
                    { binding: 1, resource: this.trailSampler },
                    { binding: 2, resource: { buffer: this.fadeFactorBuf } },
                ],
            }));
            pass.draw(3);
            pass.end();

            // 2. Trail circles (additive onto faded trail)
            const pass2 = encoder.beginRenderPass({
                colorAttachments: [{
                    view: trailOtherView,
                    loadOp: 'load',
                    storeOp: 'store',
                }],
            });
            pass2.setPipeline(this.circleTrailPipeline);
            pass2.setBindGroup(0, d.createBindGroup({
                layout: this.circleTrailPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: curBuf } },
                    { binding: 1, resource: { buffer: this.cameraBuf } },
                    { binding: 2, resource: { buffer: this.typeInfoBuf } },
                ],
            }));
            pass2.draw(6, numParticles);
            pass2.end();

            // Swap trail textures
            this.trailCurrent = this.trailCurrent === 'A' ? 'B' : 'A';
        } else {
            // Trails off — keep the current trail texture cleared so the
            // composite doesn't add a stale/frozen trail image.
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: trailCurrentView,
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.end();
        }

        // 3. Composite (background + trail)
        {
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view,
                    clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.compositePipeline);
            // Read the CURRENT trail (post-swap in the trailOn case, the just-
            // cleared one in the trailOff case). Reading the pre-swap view here
            // would show a stale trail from the previous frame.
            const currentTrailView = this.trailCurrent === 'A' ? this.trailViewA : this.trailViewB;
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.compositePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: currentTrailView },
                    { binding: 1, resource: this.trailSampler },
                    { binding: 2, resource: { buffer: this.cameraBuf } },
                    { binding: 3, resource: { buffer: this.bgColorBuf } },
                ],
            }));
            pass.draw(3);
            pass.end();
        }

        // 4. Particles (circles, normal blending)
        {
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view,
                    loadOp: 'load',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.circlePipeline);
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.circlePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: curBuf } },
                    { binding: 1, resource: { buffer: this.cameraBuf } },
                    { binding: 2, resource: { buffer: this.typeInfoBuf } },
                ],
            }));
            pass.draw(6, numParticles);
            pass.end();
        }

        // 5. Glow (additive)
        if (opts.glow && opts.glowIntensity > 0) {
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view,
                    loadOp: 'load',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.glowPipeline);
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.glowPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: curBuf } },
                    { binding: 1, resource: { buffer: this.cameraBuf } },
                    { binding: 2, resource: { buffer: this.typeInfoBuf } },
                    { binding: 3, resource: { buffer: this.glowParamsBuf } },
                ],
            }));
            pass.draw(6, numParticles);
            pass.end();
        }

        // 6. Death FX (additive)
        if (opts.deathFX && opts.deathFX.length > 0) {
            const fxCount = Math.min(opts.deathFX.length, MAX_FX);
            const fxData = new Float32Array(MAX_FX * 4);
            for (let i = 0; i < fxCount; i++) {
                const f = opts.deathFX[i];
                fxData[i * 4] = f.x;
                fxData[i * 4 + 1] = f.y;
                fxData[i * 4 + 2] = f.life;
                fxData[i * 4 + 3] = f.type;
            }
            q.writeBuffer(this.fxBuf, 0, fxData);

            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view,
                    loadOp: 'load',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.fxPipeline);
            pass.setBindGroup(0, d.createBindGroup({
                layout: this.fxPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.fxBuf } },
                    { binding: 1, resource: { buffer: this.cameraBuf } },
                    { binding: 2, resource: { buffer: this.typeInfoBuf } },
                    { binding: 3, resource: { buffer: this.glowSizeBuf } },
                ],
            }));
            pass.draw(6, fxCount);
            pass.end();
        }

        // 7. Vectors & Grid (lines)
        if (opts.showVectors || opts.showGrid) {
            const lineVerts = this._buildLineVertices(opts);
            this._lineVertexCount = lineVerts.length / 6; // 6 floats per vertex (pos vec2 + color vec4)
            if (this._lineVertexCount > 0) {
                q.writeBuffer(this.lineBuf, 0, lineVerts);

                const pass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view,
                        loadOp: 'load',
                        storeOp: 'store',
                    }],
                });
                pass.setPipeline(this.linePipeline);
                pass.setBindGroup(0, d.createBindGroup({
                    layout: this.linePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.cameraBuf } },
                    ],
                }));
                pass.setVertexBuffer(0, this.lineBuf, 0, lineVerts.byteLength);
                pass.draw(this._lineVertexCount);
                pass.end();
            }
        }

        q.submit([encoder.finish()]);
    }

    /**
     * Build line vertex data for velocity vectors and grid overlay.
     * Each line is 6 vertices (2 triangles), each vertex = (x, y, r, g, b, a) = 6 floats = 24 bytes.
     */
    _buildLineVertices(opts) {
        const verts = [];
        const p = this._params;
        if (!p) return verts;

        // Read particle data from CPU cache (set by readParticles)
        const pd = this._cpuParticles;

        if (opts.showVectors && pd) {
            const alpha = 0.25;
            for (let i = 0; i < this.maxParticles; i++) {
                const o = i * PARTICLE_FLOATS;
                if (pd[o + P_ALIVE] < 0.5) continue;
                const x0 = pd[o + P_X], y0 = pd[o + P_Y];
                const x1 = x0 + pd[o + P_VX] * 8, y1 = y0 + pd[o + P_VY] * 8;
                _pushLine(verts, x0, y0, x1, y1, 1, 1, 1, alpha);
            }
        }

        if (opts.showGrid && p) {
            const cs = p.radius; // grid cell size = interaction radius
            const alpha = 0.04;
            for (let x = 0; x <= p.worldW; x += cs) {
                _pushLine(verts, x, 0, x, p.worldH, 1, 1, 1, alpha);
            }
            for (let y = 0; y <= p.worldH; y += cs) {
                _pushLine(verts, 0, y, p.worldW, y, 1, 1, 1, alpha);
            }
        }

        return new Float32Array(verts);
    }

    /* ---- Readback ---- */

    /** The readback buffer to use this frame (alternates each frame). */
    get _rbCurrent() {
        return this._rbFlip ? this._readbackBufB : this._readbackBufA;
    }

    /**
     * Start async readback of particle data. Call after submitCompute().
     * Resolves with a Float32Array of particle data.
     */
    readParticles() {
        const buf = this._rbCurrent;
        this._rbFlip = !this._rbFlip; // advance ping-pong for the next frame
        return buf.mapAsync(GPUMapMode.READ).then(() => {
            const data = new Float32Array(buf.getMappedRange().slice(0));
            buf.unmap();
            this._cpuParticles = data;

            // Count alive particles
            let alive = 0;
            for (let i = 0; i < this.maxParticles; i++) {
                if (data[i * PARTICLE_FLOATS + P_ALIVE] >= 0.5) alive++;
            }
            this.aliveCount = alive;

            return data;
        }).catch(e => {
            // Map failed (buffer still in use) — return stale data
            return this._cpuParticles;
        });
    }

    /** Get the cached CPU-side particle data (from last readback). */
    getCPUParticles() {
        return this._cpuParticles;
    }
}

/** Push a line (as 6 vertices / 2 triangles) into the vertex array. */
function _pushLine(verts, x0, y0, x1, y1, r, g, b, a) {
    // Thin quad: 0.5px half-width perpendicular to line direction
    const dx = x1 - x0, dy = y1 - y0;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.001) return;
    const nx = -dy / len * 0.5, ny = dx / len * 0.5; // perpendicular

    const p = [
        x0 + nx, y0 + ny, r, g, b, a,
        x1 + nx, y1 + ny, r, g, b, a,
        x0 - nx, y0 - ny, r, g, b, a,
        x0 - nx, y0 - ny, r, g, b, a,
        x1 + nx, y1 + ny, r, g, b, a,
        x1 - nx, y1 - ny, r, g, b, a,
    ];
    verts.push(...p);
}
