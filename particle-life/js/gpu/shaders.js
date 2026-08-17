/**
 * WGSL shader source code for the WebGPU particle life simulation.
 *
 * Compute: binClear, binCount, prefixSum, binSort, computeAll
 * Render:  trailFade, circle, glow, composite, lineSimple, deathFX
 */

// Bump this on any WGSL change so you can confirm a fresh module actually loaded
// (browsers/dev servers can serve a stale shaders.js while webgpu-engine.js updates).
// If the trail mirror persists, check the console for this line — if it's missing,
// the tab is running old shader code: hard-refresh (Ctrl/Cmd+Shift+R).
console.info('[shaders.js] v4 — FS_QUAD_VS screen-aligned UV (trail fade Y-flip fix)');

// ====== Common types ======
const COMMON = /* wgsl */`
struct Particle {
    x: f32, y: f32, prevX: f32, prevY: f32,
    vx: f32, vy: f32, ptype: f32, energy: f32,
    age: f32, reproCooldown: f32, alive: f32, id: f32,
};

struct SimParams {
    worldW: f32, worldH: f32,
    radius: f32, maxSpeed: f32, damping: f32, dt: f32,
    wrap: f32, lifeEnabled: f32,
    energyDecay: f32, collisionCost: f32, feedRate: f32,
    reproNeighbors: f32, reproEnergy: f32, reproCooldown: f32,
    numTypes: f32, gridW: f32, gridH: f32,
    cellSize: f32, maxParticles: f32,
    trail: f32,
    _pad0: f32, _pad1: f32, _pad2: f32, _pad3: f32,
};

struct Camera {
    viewX: f32, viewY: f32, zoom: f32,
    worldW: f32, worldH: f32,
    canvasW: f32, canvasH: f32,
    _pad0: f32, _pad1: f32,
};

struct TypeInfo {
    r: f32, g: f32, b: f32, size: f32,
};

struct PSParams {
    stepSize: u32,
    count: u32,
};

// A reproduction request recorded by a parent particle, resolved into a free
// slot in a SEPARATE pass (so the parent thread and the free slot's own thread
// never both write the same output slot — that would be a write race).
struct SpawnReq {
    x: f32, y: f32, vx: f32, vy: f32,
    ptype: f32, energy: f32, age: f32, cooldown: f32,
    targetSlot: f32, valid: f32,
};
`;

// ====== Fullscreen quad vertex shader (shared) ======
const FS_QUAD_VS = /* wgsl */`
struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOut {
    let pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    )[vid];
    var out: VertexOut;
    out.pos = vec4f(pos, 0.0, 1.0);
    // Screen-aligned UV: NDC y=+1 is the TOP of the viewport, but texture v=0
    // is the top row — so v must be inverted. Without this, the trail-fade pass
    // copies the previous frame's trail vertically mirrored (the "mirrored
    // ghost in the bottom half" bug).
    out.uv = vec2f((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return out;
}
`;

// ====== Compute: binClear ======
export const BIN_CLEAR = /* wgsl */`
${COMMON}
@group(0) @binding(0) var<storage, read_write> binData: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> sortCounters: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> freeCount: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> freeHead: atomic<u32>;
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let numBins = u32(params.gridW) * u32(params.gridH);
    if (gid.x <= numBins) {
        atomicStore(&binData[gid.x], 0u);
    }
    if (gid.x < numBins) {
        atomicStore(&sortCounters[gid.x], 0u);
    }
    if (gid.x == 0u) {
        atomicStore(&freeCount, 0u);
        atomicStore(&freeHead, 0u);
    }
}
`;

// ====== Compute: binCount ======
export const BIN_COUNT = /* wgsl */`
${COMMON}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> binData: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> freeList: array<u32>;
@group(0) @binding(3) var<storage, read_write> freeCount: atomic<u32>;
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u32(params.maxParticles)) { return; }

    let p = particles[i];
    if (p.alive < 0.5) {
        let idx = atomicAdd(&freeCount, 1u);
        if (idx < u32(params.maxParticles)) {
            freeList[idx] = i;
        }
        return;
    }

    let cx = min(u32(floor(p.x / params.cellSize)), u32(params.gridW) - 1u);
    let cy = min(u32(floor(p.y / params.cellSize)), u32(params.gridH) - 1u);
    let bin = cy * u32(params.gridW) + cx;
    atomicAdd(&binData[bin + 1u], 1u);
}
`;

// ====== Compute: prefixSum ======
export const PREFIX_SUM = /* wgsl */`
${COMMON}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> ps: PSParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= ps.count) { return; }
    if (i < ps.stepSize) {
        output[i] = input[i];
    } else {
        output[i] = input[i - ps.stepSize] + input[i];
    }
}
`;

// ====== Compute: binSort ======
export const BIN_SORT = /* wgsl */`
${COMMON}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> sorted: array<Particle>;
@group(0) @binding(2) var<storage, read> binOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> sortCounters: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u32(params.maxParticles)) { return; }

    let p = particles[i];
    if (p.alive < 0.5) { return; }

    let cx = min(u32(floor(p.x / params.cellSize)), u32(params.gridW) - 1u);
    let cy = min(u32(floor(p.y / params.cellSize)), u32(params.gridH) - 1u);
    let bin = cy * u32(params.gridW) + cx;

    let offset = atomicAdd(&sortCounters[bin], 1u);
    let dest = binOffsets[bin] + offset;
    sorted[dest] = p;
}
`;

// ====== Compute: computeAll ======
// Forces + integration + life cycle in one pass.
export const COMPUTE_ALL = /* wgsl */`
${COMMON}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> outParticles: array<Particle>;
@group(0) @binding(2) var<storage, read> sorted: array<Particle>;
@group(0) @binding(3) var<storage, read> binOffsets: array<u32>;
@group(0) @binding(4) var<uniform> params: SimParams;
@group(0) @binding(5) var<storage, read> matrix: array<f32>;
@group(0) @binding(6) var<storage, read> freeList: array<u32>;
@group(0) @binding(7) var<storage, read> freeCount: atomic<u32>;
@group(0) @binding(8) var<storage, read_write> freeHead: atomic<u32>;
@group(0) @binding(9) var<storage, read_write> spawnReq: array<SpawnReq>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u32(params.maxParticles)) { return; }

    let p = particles[i];
    if (p.alive < 0.5) {
        outParticles[i] = p;
        // Write an invalid spawn request (otherwise a stale valid flag from a
        // previous frame would make SPAWN_RESOLVE respawn into this slot).
        spawnReq[i] = SpawnReq(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    let worldW = params.worldW;
    let worldH = params.worldH;
    let radius = params.radius;
    let maxSpeed = params.maxSpeed;
    let damping = params.damping;
    let dt = params.dt;
    let numTypes = u32(params.numTypes);
    let lifeOn = params.lifeEnabled > 0.5;
    let wrapOn = params.wrap > 0.5;
    let gridW = i32(params.gridW);
    let gridH = i32(params.gridH);

    let cellX = i32(min(u32(floor(p.x / params.cellSize)), u32(params.gridW) - 1u));
    let cellY = i32(min(u32(floor(p.y / params.cellSize)), u32(params.gridH) - 1u));

    var fx = 0.0;
    var fy = 0.0;
    var nCount = 0u;
    var feed = 0.0;
    var contact = 0.0;

    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            var nx = cellX + dx;
            var ny = cellY + dy;

            if (wrapOn) {
                if (nx < 0) { nx += gridW; }
                else if (nx >= gridW) { nx -= gridW; }
                if (ny < 0) { ny += gridH; }
                else if (ny >= gridH) { ny -= gridH; }
            } else {
                if (nx < 0 || nx >= gridW || ny < 0 || ny >= gridH) { continue; }
            }

            let nBin = u32(ny) * u32(gridW) + u32(nx);
            let binStart = binOffsets[nBin];
            let binEnd = binOffsets[nBin + 1u];

            for (var j = binStart; j < binEnd; j += 1u) {
                let q = sorted[j];
                if (q.alive < 0.5) { continue; }
                if (q.id == p.id) { continue; }

                let qType = u32(q.ptype);
                if (qType >= numTypes) { continue; }

                var ddx = q.x - p.x;
                var ddy = q.y - p.y;

                if (wrapOn) {
                    if (ddx > worldW * 0.5) { ddx -= worldW; }
                    else if (ddx < -worldW * 0.5) { ddx += worldW; }
                    if (ddy > worldH * 0.5) { ddy -= worldH; }
                    else if (ddy < -worldH * 0.5) { ddy += worldH; }
                }

                let dSq = ddx * ddx + ddy * ddy;
                if (dSq >= radius * radius || dSq < 0.01) { continue; }

                let d = sqrt(dSq);
                let strength = matrix[u32(p.ptype) * numTypes + qType];
                let f = strength * (1.0 - d / radius);

                fx += (ddx / d) * f;
                fy += (ddy / d) * f;

                if (lifeOn) {
                    nCount += 1u;
                    if (strength > 0.0) { feed += f; }
                    if (d < 5.0) { contact += (5.0 - d) / 5.0; }
                }

                if (d < 5.0) {
                    let sep = (5.0 - d) / 5.0 * 0.5;
                    fx -= (ddx / d) * sep * 2.0;
                    fy -= (ddy / d) * sep * 2.0;
                }
            }
        }
    }

    // Integration
    var nvx = (p.vx + fx * dt) * damping;
    var nvy = (p.vy + fy * dt) * damping;

    let spd = sqrt(nvx * nvx + nvy * nvy);
    if (spd > maxSpeed) {
        let inv = maxSpeed / spd;
        nvx *= inv;
        nvy *= inv;
    }

    var nx = p.x + nvx;
    var ny = p.y + nvy;
    var nprevX = p.prevX;
    var nprevY = p.prevY;

    if (wrapOn) {
        // "Opposite angle" wrap. Instead of a plain torus axis-wrap
        // (left->right at the SAME y, top->bottom at the SAME x), reflect the
        // particle through the canvas CENTER. The canvas center defines a
        // circle; a point at angle θ around it re-enters at θ + 180° — the
        // diametrically-opposite edge/corner. A rectangle is centrally
        // symmetric, so reflecting an out-of-bounds point through the center
        // lands on the opposite boundary. The circle only sets the angle; the
        // rectangle is still the actual canvas. Velocity is left unchanged:
        // central symmetry turns the outward velocity at the exit edge into
        // the inward velocity at the re-entry edge, so the exit angle is
        // preserved. The 0.5px inset keeps the re-entering particle strictly
        // inside so it cannot re-trigger the boundary next frame.
        const EPS = 0.5;
        if (nx < 0.0 || nx >= worldW || ny < 0.0 || ny >= worldH) {
            nx = clamp(worldW - nx, EPS, worldW - EPS);
            ny = clamp(worldH - ny, EPS, worldH - EPS);
            nprevX = clamp(worldW - nprevX, EPS, worldW - EPS);
            nprevY = clamp(worldH - nprevY, EPS, worldH - EPS);
        }
    } else {
        if (nx < 0.0) { nx = 0.0; nvx = -nvx; }
        else if (nx >= worldW) { nx = worldW; nvx = -nvx; }
        if (ny < 0.0) { ny = 0.0; nvy = -nvy; }
        else if (ny >= worldH) { ny = worldH; nvy = -nvy; }
    }

    // Life cycle
    var newEnergy = p.energy;
    var newAlive = 1.0;
    var newAge = p.age + dt;
    var newCooldown = max(0.0, p.reproCooldown - dt);

    // Spawn request (defaults to invalid — written every frame, resolved in a
    // separate pass so no two threads ever write the same output slot).
    var reqX = nx;
    var reqY = ny;
    var reqVX = 0.0;
    var reqVY = 0.0;
    var reqPType = p.ptype;
    var reqEnergy = 0.0;
    var reqTarget = 0.0;
    var reqValid = 0.0;

    if (lifeOn) {
        newEnergy = p.energy + (params.feedRate * feed - params.energyDecay - params.collisionCost * contact) * dt;
        if (newEnergy > 1.0) { newEnergy = 1.0; }

        if (newEnergy <= 0.0) {
            newAlive = 0.0;
            // Newly-dead slots enter the free list on the NEXT frame's binCount.
        } else if (nCount >= u32(params.reproNeighbors) && newEnergy >= params.reproEnergy && newCooldown <= 0.0) {
            let hIdx = atomicAdd(&freeHead, 1u);
            let fc = atomicLoad(&freeCount);
            if (hIdx < fc) {
                let slot = freeList[hIdx];
                let parentEnergy = newEnergy * 0.5;
                // Deterministic pseudo-random jitter based on slot index
                let a = f32(slot % 628u) / 100.0 * 6.2831853;
                let s = f32((slot * 7u + 3u) % 50u) / 100.0;
                reqX = nx + cos(a) * s * 10.0;
                reqY = ny + sin(a) * s * 10.0;
                if (wrapOn) {
                    if (reqX < 0.0) { reqX += worldW; }
                    else if (reqX >= worldW) { reqX -= worldW; }
                    if (reqY < 0.0) { reqY += worldH; }
                    else if (reqY >= worldH) { reqY -= worldH; }
                }
                reqVX = cos(a) * s;
                reqVY = sin(a) * s;
                reqPType = p.ptype;
                reqEnergy = parentEnergy;
                reqTarget = f32(slot);
                reqValid = 1.0;
                newEnergy = parentEnergy;
            }
        }
    }

    spawnReq[i] = SpawnReq(reqX, reqY, reqVX, reqVY, reqPType, reqEnergy, 0.0, params.reproCooldown, reqTarget, reqValid);

    outParticles[i] = Particle(
        nx, ny, nprevX, nprevY,
        nvx, nvy, p.ptype, newEnergy,
        newAge, newCooldown, newAlive, p.id
    );
}
`;

// ====== Compute: spawnResolve ======
// Materializes pending spawn requests into free slots. Runs AFTER computeAll in
// its own dispatch: each free slot is written by exactly one thread, so there
// is no write race. The child's id is the slot index (guaranteed unique because
// the slot was dead when the request was made).
export const SPAWN_RESOLVE = /* wgsl */`
${COMMON}
@group(0) @binding(0) var<storage, read> spawnReq: array<SpawnReq>;
@group(0) @binding(1) var<storage, read_write> outParticles: array<Particle>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u32(params.maxParticles)) { return; }
    let req = spawnReq[i];
    if (req.valid < 0.5) { return; }
    let slot = u32(req.targetSlot);
    outParticles[slot] = Particle(
        req.x, req.y, req.x, req.y,
        req.vx, req.vy, req.ptype, req.energy,
        0.0, params.reproCooldown, 1.0, req.targetSlot
    );
}
`;

// ====== Render: trailFade ======
// Fullscreen quad that fades the trail texture by a factor.
export const TRAIL_FADE = /* wgsl */`
${FS_QUAD_VS}

@group(0) @binding(0) var trailTex: texture_2d<f32>;
@group(0) @binding(1) var trailSampler: sampler;
@group(0) @binding(2) var<uniform> fadeFactor: f32;

@fragment
fn fs(v: VertexOut) -> @location(0) vec4f {
    let color = textureSample(trailTex, trailSampler, v.uv).rgb;
    return vec4f(color * fadeFactor, 1.0);
}
`;

// ====== Render: circle (particles & trail dots) ======
export const CIRCLE = /* wgsl */`
${COMMON}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) offset: vec2f,
    @location(1) color: vec4f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> typeInfo: array<TypeInfo, 16>;

const corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
);

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOut {
    let p = particles[iid];
    var out: VertexOut;

    if (p.alive < 0.5) {
        out.pos = vec4f(-10000.0, -10000.0, 0.0, 1.0);
        out.offset = vec2f(0.0);
        out.color = vec4f(0.0);
        return out;
    }

    let c = corners[vid];
    let ti = typeInfo[u32(p.ptype)];
    let worldPos = vec2f(p.x, p.y) + c * ti.size;
    let screenPos = (worldPos - vec2f(camera.viewX, camera.viewY)) * camera.zoom;

    out.pos = vec4f(
        screenPos.x * (2.0 / camera.worldW) - 1.0,
        1.0 - screenPos.y * (2.0 / camera.worldH),
        0.0, 1.0);
    out.offset = c;
    out.color = vec4f(ti.r, ti.g, ti.b, 1.0);
    return out;
}

@fragment
fn fs(v: VertexOut) -> @location(0) vec4f {
    let d = length(v.offset);
    if (d > 1.0) { discard; }
    let aa = 1.5 / max(1.0, camera.zoom);
    let alpha = smoothstep(1.0, 1.0 - aa, d);
    return vec4f(v.color.rgb, alpha);
}
`;

// ====== Render: circle for the trail texture (world-locked) ======
// Same disc as CIRCLE but drawn in WORLD space into the world-sized trail
// texture (independent of the camera), so accumulated trails stay fixed in
// world space while zooming/panning. The composite applies the camera when
// reading the trail back to the screen.
export const CIRCLE_TRAIL = /* wgsl */`
${COMMON}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) offset: vec2f,
    @location(1) color: vec4f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> typeInfo: array<TypeInfo, 16>;

const corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
);

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOut {
    let p = particles[iid];
    var out: VertexOut;

    if (p.alive < 0.5) {
        out.pos = vec4f(-10000.0, -10000.0, 0.0, 1.0);
        out.offset = vec2f(0.0);
        out.color = vec4f(0.0);
        return out;
    }

    let c = corners[vid];
    let ti = typeInfo[u32(p.ptype)];
    let worldPos = vec2f(p.x, p.y) + c * ti.size;

    // World -> NDC over the world-sized trail texture (no camera).
    // NOTE the Y flip (1.0 - ...) so the trail texture is stored in the same
    // top-left, y-down orientation as the world/canvas. COMPOSITE reads it back
    // with uv = worldPos/(worldW,worldH) (no flip) and the screen-space
    // particles (CIRCLE) are also y-flipped. Without this flip the trail is
    // stored vertically mirrored and appears as a mirrored ghost behind the
    // correctly-oriented particles.
    out.pos = vec4f(
        worldPos.x * (2.0 / camera.worldW) - 1.0,
        1.0 - worldPos.y * (2.0 / camera.worldH),
        0.0, 1.0);
    out.offset = c;
    out.color = vec4f(ti.r, ti.g, ti.b, 1.0);
    return out;
}

@fragment
fn fs(v: VertexOut) -> @location(0) vec4f {
    let d = length(v.offset);
    if (d > 1.0) { discard; }
    let aa = 0.2;
    let alpha = smoothstep(1.0, 1.0 - aa, d);
    // Premultiply the disc color by its alpha so the additive (one,one) blend
    // lays down a soft, color-faithful trail instead of a hard full-color disc
    // that blows out to white where particles cluster.
    return vec4f(v.color.rgb * alpha, alpha);
}
`;

// ====== Render: glow ======
export const GLOW = /* wgsl */`
${COMMON}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) offset: vec2f,
    @location(1) color: vec4f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> typeInfo: array<TypeInfo, 16>;
@group(0) @binding(3) var<uniform> glowParams: vec2f;

const corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
);

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOut {
    let p = particles[iid];
    var out: VertexOut;

    if (p.alive < 0.5) {
        out.pos = vec4f(-10000.0, -10000.0, 0.0, 1.0);
        out.offset = vec2f(0.0);
        out.color = vec4f(0.0);
        return out;
    }

    let c = corners[vid];
    let ti = typeInfo[u32(p.ptype)];
    let size = ti.size * glowParams.x;
    let worldPos = vec2f(p.x, p.y) + c * size;
    let screenPos = (worldPos - vec2f(camera.viewX, camera.viewY)) * camera.zoom;

    out.pos = vec4f(
        screenPos.x * (2.0 / camera.worldW) - 1.0,
        1.0 - screenPos.y * (2.0 / camera.worldH),
        0.0, 1.0);
    out.offset = c;
    out.color = vec4f(ti.r, ti.g, ti.b, glowParams.y);
    return out;
}

@fragment
fn fs(v: VertexOut) -> @location(0) vec4f {
    let d = length(v.offset);
    if (d > 1.0) { discard; }
    let alpha = exp(-6.0 * d * d) / 64.0;
    return vec4f(v.color.rgb * alpha, alpha);
}
`;

// ====== Render: composite (background + trail) ======
export const COMPOSITE = /* wgsl */`
${COMMON}

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) fragPos: vec2f,
};

@group(0) @binding(0) var trailTex: texture_2d<f32>;
@group(0) @binding(1) var trailSampler: sampler;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<uniform> bgColor: vec4f;

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOut {
    let pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    )[vid];
    var out: VertexOut;
    out.pos = vec4f(pos, 0.0, 1.0);
    out.fragPos = (pos + 1.0) * 0.5 * vec2f(camera.canvasW, camera.canvasH);
    return out;
}

@fragment
fn fs(v: VertexOut) -> @location(0) vec4f {
    // v.pos.xy is the fragment position in PHYSICAL (device-pixel) canvas
    // coords, but world/camera math is in LOGICAL (CSS) units. Convert
    // physical -> logical via worldSize / physicalCanvasSize (= 1/dpr) so the
    // trail lines up with the screen-space particles at every pixel ratio.
    let logicalPx = v.pos.xy * (vec2f(camera.worldW, camera.worldH) / vec2f(camera.canvasW, camera.canvasH));
    let worldPos = logicalPx / camera.zoom + vec2f(camera.viewX, camera.viewY);
    let uv = worldPos / vec2f(camera.worldW, camera.worldH);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4f(bgColor.rgb, 1.0);
    }
    let trail = textureSample(trailTex, trailSampler, uv).rgb;
    return vec4f(bgColor.rgb + trail, 1.0);
}
`;

// ====== Render: lineSimple (velocity vectors & grid) ======
// Pre-computed line quad vertices from CPU.
// Vertex layout: pos (vec2f) + color (vec4f) = 24 bytes per vertex
export const LINE_SIMPLE = /* wgsl */`
struct VertexIn {
    @location(0) pos: vec2f,
    @location(1) color: vec4f,
};

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
};

struct Camera_simple {
    viewX: f32, viewY: f32, zoom: f32,
    worldW: f32, worldH: f32,
    canvasW: f32, canvasH: f32,
    _pad0: f32, _pad1: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera_simple;

@vertex
fn vs(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let screenPos = (v.pos - vec2f(cam.viewX, cam.viewY)) * cam.zoom;
    out.pos = vec4f(
        screenPos.x * (2.0 / cam.worldW) - 1.0,
        1.0 - screenPos.y * (2.0 / cam.worldH),
        0.0, 1.0);
    out.color = v.color;
    return out;
}

@fragment
fn fs(v: VertexOut) -> @location(0) vec4f {
    return v.color;
}
`;

// ====== Render: deathFX ======
// Instance data: (x, y, life, type) = 16 bytes per instance
export const DEATH_FX = /* wgsl */`
${COMMON}

struct FXInstance {
    x: f32, y: f32, life: f32, ptype: f32,
};

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) offset: vec2f,
    @location(1) color: vec4f,
};

@group(0) @binding(0) var<storage, read> fxInstances: array<FXInstance>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> typeInfo: array<TypeInfo, 16>;
@group(0) @binding(3) var<uniform> glowSize: f32;

const corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
);

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOut {
    let fx = fxInstances[iid];
    var out: VertexOut;

    if (fx.life <= 0.0) {
        out.pos = vec4f(-10000.0, -10000.0, 0.0, 1.0);
        out.offset = vec2f(0.0);
        out.color = vec4f(0.0);
        return out;
    }

    let c = corners[vid];
    let ti = typeInfo[u32(fx.ptype)];
    let size = ti.size * glowSize * fx.life;
    let worldPos = vec2f(fx.x, fx.y) + c * size;
    let screenPos = (worldPos - vec2f(camera.viewX, camera.viewY)) * camera.zoom;

    out.pos = vec4f(
        screenPos.x * (2.0 / camera.worldW) - 1.0,
        1.0 - screenPos.y * (2.0 / camera.worldH),
        0.0, 1.0);
    out.offset = c;
    let alpha = min(0.6, fx.life * 0.5);
    out.color = vec4f(ti.r, ti.g, ti.b, alpha);
    return out;
}

@fragment
fn fs(v: VertexOut) -> @location(0) vec4f {
    let d = length(v.offset);
    if (d > 1.0) { discard; }
    let alpha = exp(-6.0 * d * d) / 64.0;
    return vec4f(v.color.rgb * alpha, alpha);
}
`;
