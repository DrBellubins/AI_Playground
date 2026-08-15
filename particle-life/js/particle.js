/**
 * Particle — a single entity in the simulation with position, velocity, and type.
 */
export class Particle {
  constructor(x, y, type, index) {
    this.x = x;
    this.y = y;
    this.prevX = x;
    this.prevY = y;
    this.vx = 0;
    this.vy = 0;
    this.type = type;
    this._index = index;

    // Life cycle (reproduction & death) — see Simulation.lifeStep()
    this.energy = 1;        // 0..1 — particle dies when this reaches 0
    this.age = 0;           // seconds alive
    this.reproCooldown = 0; // seconds until this particle may split again
  }
}
