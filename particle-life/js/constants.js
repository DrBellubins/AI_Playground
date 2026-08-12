/**
 * Default particle type definitions and preset interaction matrices.
 */

export const DEFAULT_TYPES = [
  { name: 'Red',    color: '#ff4466', size: 2.5, countPct: 20 },
  { name: 'Green',  color: '#44dd77', size: 2.0, countPct: 20 },
  { name: 'Blue',   color: '#4488ff', size: 3.0, countPct: 20 },
  { name: 'Yellow', color: '#ffcc33', size: 1.8, countPct: 20 },
  { name: 'Purple', color: '#bb55ff', size: 2.2, countPct: 20 },
];

/**
 * Build an NxN matrix from a generator function.
 */
function matrixFromFn(n, fn) {
  const m = [];
  for (let i = 0; i < n; i++) {
    m[i] = [];
    for (let j = 0; j < n; j++) m[i][j] = fn(i, j);
  }
  return m;
}

/**
 * Preset interaction matrices — each returns an NxN array of forces.
 */
export const PRESETS = {
  swirl(n) {
    return matrixFromFn(n, (i, j) => {
      if (i === j) return 0;
      const d = ((j - i + n) % n) / n;
      return d < 0.25 ? 1.5 : d < 0.5 ? -0.5 : d < 0.75 ? -1.5 : 0.5;
    });
  },
  clusters(n) {
    return matrixFromFn(n, (i, j) => i === j ? 1.2 : -1.0);
  },
  chaos(_n) {
    return matrixFromFn(5, () => (Math.random() * 4 - 2));
  },
  galaxy(n) {
    return matrixFromFn(n, (i, j) => {
      if (i === j) return 0.8;
      const d = Math.abs(i - j) / n;
      return d < 0.15 ? 1.5 : d < 0.3 ? -1.0 : 0.3;
    });
  },
  random(n) {
    return matrixFromFn(n, () => +(Math.random() * 4 - 2).toFixed(2));
  },
};
