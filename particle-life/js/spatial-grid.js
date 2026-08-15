/**
 * SpatialGrid — spatial partitioning for fast neighbor lookups.
 * Reduces the O(n²) force computation to roughly O(n) by only checking nearby particles.
 *
 * Uses a counting-sort (CSR) bucket layout over flat typed arrays instead of
 * a Map of arrays:
 *   - zero per-frame allocations after warm-up
 *   - no Map hashing on build or query
 *   - deterministic bucket order (ascending particle index), so consumers
 *     see the same neighbor order as the previous Map-based implementation
 *
 * With cellSize == interaction radius, a neighbor query scans the 3x3 block
 * of cells around the query particle.
 */
export class SpatialGrid {
  constructor(cellSize) {
    this.cellSize = cellSize;
    this.cols = 0;
    this.rows = 0;

    // Allocated/reallocated only when the grid dimensions or capacity change.
    this.counts = null;   // Uint32Array(cols*rows)
    this.offsets = null;  // Uint32Array(cols*rows+1) — cell c owns items[offsets[c]..offsets[c+1])
    this.cursor = null;   // Uint32Array(cols*rows) — scratch, reuse of counts during scatter
    this.items = null;    // Uint32Array(capacity) — particle indices, sorted by cell
    this.capacity = 0;
  }

  /** Rebuild buckets from current particle positions (call once per physics step). */
  build(particles, w, h) {
    const cs = this.cellSize;
    const cols = Math.max(1, Math.ceil(w / cs));
    const rows = Math.max(1, Math.ceil(h / cs));
    const n = particles.length;

    if (cols !== this.cols || rows !== this.rows) {
      this.cols = cols;
      this.rows = rows;
      const cells = cols * rows;
      this.counts = new Uint32Array(cells);
      this.offsets = new Uint32Array(cells + 1);
      this.cursor = new Uint32Array(cells);
    }
    if (n > this.capacity) {
      this.capacity = Math.max(n, this.capacity * 2, 1024);
      this.items = new Uint32Array(this.capacity);
    }

    const { counts, offsets, cursor, items } = this;
    const c = cols, r = rows;

    counts.fill(0);

    // Pass 1: count per cell
    for (let i = 0; i < n; i++) {
      const p = particles[i];
      let cx = Math.floor(p.x / cs); if (cx >= c) cx = c - 1;
      let cy = Math.floor(p.y / cs); if (cy >= r) cy = r - 1;
      counts[cy * c + cx]++;
    }

    // Pass 2: prefix sum
    let sum = 0;
    const cellCount = c * r;
    for (let i = 0; i < cellCount; i++) {
      offsets[i] = sum;
      sum += counts[i];
      cursor[i] = offsets[i];
    }
    offsets[cellCount] = sum;

    // Pass 3: scatter
    for (let i = 0; i < n; i++) {
      const p = particles[i];
      let cx = Math.floor(p.x / cs); if (cx >= c) cx = c - 1;
      let cy = Math.floor(p.y / cs); if (cy >= r) cy = r - 1;
      items[cursor[cy * c + cx]++] = i;
    }
  }

  /**
   * Visit every particle within `radius` of `particles[index]` (excluding
   * itself). Zero-allocation: `fn(indexOfNeighbor)` is invoked per neighbor
   * instead of building a result array.
   *
   * Visit order matches the previous getNeighbors() exactly: cell offsets
   * scanned (ox outer, oy inner) from -r..+r, buckets in ascending index
   * order.
   */
  forEachNeighbor(index, particle, radius, fn) {
    const cs = this.cellSize;
    const r = Math.ceil(radius / cs);
    const pcx = Math.floor(particle.x / cs);
    const pcy = Math.floor(particle.y / cs);
    const { cols, rows, offsets, items } = this;

    for (let ox = -r; ox <= r; ox++) {
      const cx = pcx + ox;
      if (cx < 0 || cx >= cols) continue;
      for (let oy = -r; oy <= r; oy++) {
        const cy = pcy + oy;
        if (cy < 0 || cy >= rows) continue;
        const c = cy * cols + cx;
        const start = offsets[c];
        const end = offsets[c + 1];
        for (let k = start; k < end; k++) {
          const qi = items[k];
          if (qi === index) continue;
          fn(qi);
        }
      }
    }
  }
}
