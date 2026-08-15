/**
 * SpatialGrid — hash-based spatial partitioning for efficient neighbor lookups.
 * Reduces the O(n²) force computation to roughly O(n) by only checking nearby particles.
 */
export class SpatialGrid {
  constructor(cellSize) {
    this.cellSize = cellSize;
    this.cells = new Map();
  }

  clear() {
    this.cells.clear();
  }

  _key(cx, cy) {
    return (cx + 1000) * 10000 + (cy + 1000);
  }

  insert(particle, cx, cy) {
    const key = this._key(cx, cy);
    let bucket = this.cells.get(key);
    if (!bucket) {
      bucket = [];
      this.cells.set(key, bucket);
    }
    bucket.push(particle);
  }

  build(particles, w, h) {
    this.clear();
    const cs = this.cellSize;
    const cols = Math.ceil(w / cs);
    const rows = Math.ceil(h / cs);
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const cx = Math.floor(p.x / cs);
      const cy = Math.floor(p.y / cs);
      this.insert(p, cx, cy);
    }
  }

  getNeighbors(particle, radius) {
    const cs = this.cellSize;
    const r = Math.ceil(radius / cs);
    const pcx = Math.floor(particle.x / cs);
    const pcy = Math.floor(particle.y / cs);
    const result = [];
    for (let dx = -r; dx <= r; dx++) {
      for (let dy = -r; dy <= r; dy++) {
        const key = this._key(pcx + dx, pcy + dy);
        const bucket = this.cells.get(key);
        if (bucket) {
          for (let i = 0; i < bucket.length; i++) {
            const q = bucket[i];
            if (q !== particle) result.push(q);
          }
        }
      }
    }
    return result;
  }
}
