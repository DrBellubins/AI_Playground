/**
 * Seeded PRNG — Mulberry32
 * Fast, deterministic, 32-bit PRNG suitable for reproducible simulations.
 */
export function mulberry32(seed) {
  let s = seed | 0;
  return function() {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
