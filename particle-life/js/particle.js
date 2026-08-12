/**
 * Particle — a single entity in the simulation with position, velocity, and type.
 */
export class Particle {
  constructor(x, y, type) {
    this.x = x;
    this.y = y;
    this.vx = 0;
    this.vy = 0;
    this.type = type;
  }
}
