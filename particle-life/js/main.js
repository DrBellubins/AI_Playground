/**
 * Main entry point — bootstraps the simulation.
 */
import { Simulation } from './simulation.js';
import { UIController } from './ui.js';

const canvas = document.getElementById('sim');
const sim = new Simulation(canvas);
const ui = new UIController(sim);
globalThis.ui = ui;
sim.start();
