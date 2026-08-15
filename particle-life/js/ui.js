import { PRESETS } from './constants.js';

/**
 * UIController — binds all DOM controls to the simulation.
 * Handles sliders, buttons, type list, interaction matrix, and export/import.
 */
export class UIController {
  constructor(sim) {
    this.sim = sim;

    // Bind simulation controls
    this.bind('s-total', 'v-total', v => {
      this.sim.totalParticles = +v;
      this.sim.adjustParticleCount();
    });
    this.bind('s-radius', 'v-radius', v => { this.sim.interactionRadius = +v; });
    this.bind('s-speed', 'v-speed', v => { this.sim.maxSpeed = +v; });
    this.bind('s-damping', 'v-damping', v => { this.sim.damping = +v; });
    this.bind('s-seed', 'v-seed', v => { this.sim.seed = +v; });

    // Play/Pause
    document.getElementById('btn-play').addEventListener('click', () => {
      this.sim.running = !this.sim.running;
      document.getElementById('btn-play').textContent = this.sim.running ? '⏸ Pause' : '▶ Play';
    });

    // Sound toggle
    document.getElementById('btn-sound').addEventListener('click', () => {
      this.sim.soundEnabled = !this.sim.soundEnabled;
      const btn = document.getElementById('btn-sound');
      if (this.sim.soundEnabled) {
        this.sim.sound.enable();
        btn.textContent = '🔊 Sound';
        btn.classList.add('active');
      } else {
        btn.textContent = '🔇 Sound';
        btn.classList.remove('active');
      }
    });

    // Reset
    document.getElementById('btn-reset').addEventListener('click', () => {
      this.sim.initParticles();
    });

    // Randomize seed + interaction matrix
    document.getElementById('btn-random-seed').addEventListener('click', () => {
      this.sim.seed = Math.floor(Math.random() * 99999);
      document.getElementById('s-seed').value = this.sim.seed;
      document.getElementById('v-seed').textContent = this.sim.seed;

      const n = this.sim.types.length;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            this.sim.matrix[i][j] = +(Math.random() * 4 - 2).toFixed(2);
          }
        }
      }

      this.sim.initParticles();
      this.buildMatrix();
    });

    // Build particle type rows
    this.buildTypeList();

    // Build matrix table
    this.buildMatrix();

    // Visuals
    this.bind('s-trail', 'v-trail', v => { this.sim.trail = +v; });
    this.bind('s-glowsize', 'v-glowsize', v => { this.sim.glowSize = +v; });
    this.bind('s-glowint', 'v-glowint', v => { this.sim.glowIntensity = +v; });
    document.getElementById('c-bg').addEventListener('input', e => {
      this.sim.bgColor = e.target.value;
    });
    document.getElementById('c-vectors').addEventListener('change', e => {
      this.sim.showVectors = e.target.checked;
    });
    document.getElementById('c-grid').addEventListener('change', e => {
      this.sim.showGrid = e.target.checked;
    });
    document.getElementById('c-glow').addEventListener('change', e => {
      this.sim.glow = e.target.checked;
    });

    // Life & death
    document.getElementById('c-life').addEventListener('change', e => {
      this.sim.lifeEnabled = e.target.checked;
    });
    this.bind('s-decay', 'v-decay', v => { this.sim.energyDecay = +v; });
    this.bind('s-collision', 'v-collision', v => { this.sim.collisionCost = +v; });
    this.bind('s-feed', 'v-feed', v => { this.sim.feedRate = +v; });
    this.bind('s-repro-n', 'v-repro-n', v => { this.sim.reproNeighbors = +v; });
    this.bind('s-repro-e', 'v-repro-e', v => { this.sim.reproEnergy = +v; });
    this.bind('s-maxp', 'v-maxp', v => { this.sim.maxParticles = +v; });
    this.bind('s-cooldown', 'v-cooldown', v => { this.sim.reproCooldown = +v; });

    // Preset buttons
    document.querySelectorAll('[data-preset]').forEach(btn => {
      btn.addEventListener('click', () => {
        const name = btn.dataset.preset;
        const fn = PRESETS[name];
        if (fn) {
          this.sim.matrix = fn(this.sim.types.length);
          this.buildMatrix();
        }
      });
    });

    // Copy matrix JSON
    document.getElementById('btn-copy-matrix').addEventListener('click', () => {
      const json = JSON.stringify(this.sim.matrix, null, 2);
      navigator.clipboard.writeText(json).then(() => {
        const btn = document.getElementById('btn-copy-matrix');
        const orig = btn.textContent;
        btn.textContent = '✓ Copied!';
        setTimeout(() => btn.textContent = orig, 1200);
      });
    });

    // Export
    document.getElementById('btn-export').addEventListener('click', () => {
      const cfg = this.sim.exportConfig();
      const json = JSON.stringify(cfg, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'particle-life-config.json';
      a.click();
      URL.revokeObjectURL(url);
    });

    // Import from textarea
    document.getElementById('btn-import').addEventListener('click', () => {
      try {
        const cfg = JSON.parse(document.getElementById('import-area').value);
        this.sim.importConfig(cfg);
      } catch (e) {
        alert('Invalid JSON: ' + e.message);
      }
    });

    // Load file
    document.getElementById('btn-load-file').addEventListener('click', () => {
      document.getElementById('file-input').click();
    });
    document.getElementById('file-input').addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const cfg = JSON.parse(reader.result);
          this.sim.importConfig(cfg);
        } catch (err) {
          alert('Invalid JSON file: ' + err.message);
        }
      };
      reader.readAsText(file);
      e.target.value = '';
    });

    // Save preset
    document.getElementById('btn-save-preset').addEventListener('click', () => {
      const name = document.getElementById('preset-name').value.trim() || 'Preset';
      const cfg = this.sim.exportConfig();
      const presets = JSON.parse(localStorage.getItem('pl_presets') || '{}');
      presets[name] = cfg;
      localStorage.setItem('pl_presets', JSON.stringify(presets));
      this.loadSavedPresets();
      const btn = document.getElementById('btn-save-preset');
      const orig = btn.textContent;
      btn.textContent = '✓ Saved!';
      setTimeout(() => btn.textContent = orig, 1200);
    });

    this.loadSavedPresets();

    // Panel toggle
    document.getElementById('panel-toggle').addEventListener('click', () => {
      document.getElementById('controls').classList.toggle('open');
    });

    // Resize
    window.addEventListener('resize', () => {
      this.sim.resize();
    });

    this.syncAll();
  }

  /** Generic slider binding: slider → value display → callback. */
  bind(sliderId, valId, onChange) {
    const slider = document.getElementById(sliderId);
    const val = document.getElementById(valId);
    slider.addEventListener('input', () => {
      val.textContent = slider.value;
      onChange(slider.value);
    });
  }

  /* ---- Build type list ---- */
  buildTypeList() {
    this.syncTypeListDOM();
  }

  syncTypeListDOM() {
    const container = document.getElementById('type-list');
    container.innerHTML = '';
    const sim = this.sim;

    sim.types.forEach((t, idx) => {
      const row = document.createElement('div');
      row.className = 'type-row';

      // Swatch
      const swatch = document.createElement('div');
      swatch.className = 'type-swatch';
      swatch.style.background = t.color;
      row.appendChild(swatch);

      // Name
      const nameInput = document.createElement('input');
      nameInput.type = 'text';
      nameInput.className = 'type-name-input';
      nameInput.value = t.name;
      nameInput.addEventListener('input', () => { sim.types[idx].name = nameInput.value; });
      row.appendChild(nameInput);

      // Size
      const sizeLabel = document.createElement('label');
      sizeLabel.textContent = 'Sz';
      sizeLabel.style.cssText = 'font-size:10px;color:rgba(255,255,255,0.5);flex-shrink:0';
      row.appendChild(sizeLabel);

      const sizeSlider = document.createElement('input');
      sizeSlider.type = 'range';
      sizeSlider.min = '1';
      sizeSlider.max = '6';
      sizeSlider.step = '0.1';
      sizeSlider.value = t.size;
      sizeSlider.style.width = '50px';
      sizeSlider.addEventListener('input', () => {
        sim.types[idx].size = +sizeSlider.value;
      });
      row.appendChild(sizeSlider);

      const sizeVal = document.createElement('span');
      sizeVal.className = 'val';
      sizeVal.style.cssText = 'flex:0 0 30px;text-align:right;font-size:10px;color:rgba(255,255,255,0.4)';
      sizeVal.textContent = t.size.toFixed(1);
      sizeSlider.addEventListener('input', () => {
        sizeVal.textContent = (+sizeSlider.value).toFixed(1);
      });
      row.appendChild(sizeVal);

      // Count %
      const pctLabel = document.createElement('label');
      pctLabel.textContent = '%';
      pctLabel.style.cssText = 'font-size:10px;color:rgba(255,255,255,0.5);flex-shrink:0';
      row.appendChild(pctLabel);

      const pctSlider = document.createElement('input');
      pctSlider.type = 'range';
      pctSlider.min = '1';
      pctSlider.max = '100';
      pctSlider.step = '1';
      pctSlider.value = t.countPct;
      pctSlider.style.width = '60px';
      pctSlider.addEventListener('input', () => {
        sim.types[idx].countPct = +pctSlider.value;
      });
      row.appendChild(pctSlider);

      const pctVal = document.createElement('span');
      pctVal.className = 'val';
      pctVal.style.cssText = 'flex:0 0 30px;text-align:right;font-size:10px;color:rgba(255,255,255,0.4)';
      pctVal.textContent = t.countPct;
      pctSlider.addEventListener('input', () => {
        pctVal.textContent = pctSlider.value;
      });
      row.appendChild(pctVal);

      container.appendChild(row);
    });
  }

  /* ---- Build matrix table ---- */
  buildMatrix() {
    const table = document.getElementById('matrix-table');
    const sim = this.sim;
    const types = sim.types;
    const matrix = sim.matrix;
    const n = types.length;

    let html = '<thead><tr><th></th>';
    for (let j = 0; j < n; j++) {
      html += `<th><span class="dot" style="background:${types[j].color}"></span></th>`;
    }
    html += '</tr></thead><tbody>';

    for (let i = 0; i < n; i++) {
      html += `<tr><th><span class="dot" style="background:${types[i].color}"></span></th>`;
      for (let j = 0; j < n; j++) {
        const val = (matrix[i] && matrix[i][j] !== undefined) ? matrix[i][j] : 0;
        if (i === j) {
          html += `<td class="self-cell">—</td>`;
        } else {
          html += `<td>
            <div class="matrix-cell">
              <input type="range" min="-2" max="2" step="0.01" value="${val}"
                style="accent-color:${val >= 0 ? '#44dd77' : '#ff4466'}"
                data-ri="${i}" data-ci="${j}">
              <input type="number" min="-2" max="2" step="0.01" value="${val.toFixed(2)}"
                data-ri="${i}" data-ci="${j}" step="0.01">
            </div>
          </td>`;
        }
      }
      html += '</tr>';
    }
    html += '</tbody>';
    table.innerHTML = html;

    // Bind matrix inputs
    table.querySelectorAll('input[type="range"]').forEach(input => {
      input.addEventListener('input', () => {
        const ri = +input.dataset.ri;
        const ci = +input.dataset.ci;
        const v = +input.value;
        matrix[ri][ci] = v;
        table.querySelector(`input[data-ri="${ri}"][data-ci="${ci}"][type="number"]`).value = v.toFixed(2);
      });
    });

    table.querySelectorAll('input[type="number"]').forEach(input => {
      input.addEventListener('input', () => {
        const ri = +input.dataset.ri;
        const ci = +input.dataset.ci;
        let v = parseFloat(input.value);
        if (isNaN(v)) v = 0;
        v = Math.max(-2, Math.min(2, v));
        matrix[ri][ci] = v;
        table.querySelector(`input[data-ri="${ri}"][data-ci="${ci}"][type="range"]`).value = v;
      });
    });
  }

  /* ---- Sync all UI to simulation ---- */
  syncAll() {
    document.getElementById('s-total').value = this.sim.totalParticles;
    document.getElementById('v-total').textContent = this.sim.totalParticles;
    document.getElementById('s-radius').value = this.sim.interactionRadius;
    document.getElementById('v-radius').textContent = this.sim.interactionRadius;
    document.getElementById('s-speed').value = this.sim.maxSpeed;
    document.getElementById('v-speed').textContent = this.sim.maxSpeed;
    document.getElementById('s-damping').value = this.sim.damping;
    document.getElementById('v-damping').textContent = this.sim.damping.toFixed(3);
    document.getElementById('s-seed').value = this.sim.seed;
    document.getElementById('v-seed').textContent = this.sim.seed;
    document.getElementById('s-trail').value = this.sim.trail;
    document.getElementById('v-trail').textContent = this.sim.trail;
    document.getElementById('c-glow').checked = this.sim.glow;
    document.getElementById('s-glowsize').value = this.sim.glowSize;
    document.getElementById('v-glowsize').textContent = this.sim.glowSize;
    document.getElementById('s-glowint').value = this.sim.glowIntensity;
    document.getElementById('v-glowint').textContent = this.sim.glowIntensity.toFixed(2);
    document.getElementById('c-bg').value = this.sim.bgColor;
    document.getElementById('c-vectors').checked = this.sim.showVectors;
    document.getElementById('c-grid').checked = this.sim.showGrid;

    // Life & death
    document.getElementById('c-life').checked = this.sim.lifeEnabled;
    document.getElementById('s-decay').value = this.sim.energyDecay;
    document.getElementById('v-decay').textContent = this.sim.energyDecay.toFixed(3);
    document.getElementById('s-collision').value = this.sim.collisionCost;
    document.getElementById('v-collision').textContent = this.sim.collisionCost.toFixed(2);
    document.getElementById('s-feed').value = this.sim.feedRate;
    document.getElementById('v-feed').textContent = this.sim.feedRate.toFixed(3);
    document.getElementById('s-repro-n').value = this.sim.reproNeighbors;
    document.getElementById('v-repro-n').textContent = this.sim.reproNeighbors;
    document.getElementById('s-repro-e').value = this.sim.reproEnergy;
    document.getElementById('v-repro-e').textContent = this.sim.reproEnergy.toFixed(2);
    document.getElementById('s-maxp').value = this.sim.maxParticles;
    document.getElementById('v-maxp').textContent = this.sim.maxParticles;
    document.getElementById('s-cooldown').value = this.sim.reproCooldown;
    document.getElementById('v-cooldown').textContent = this.sim.reproCooldown;

    this.syncTypeListDOM();
    this.buildMatrix();

    // Sound button state
    const soundBtn = document.getElementById('btn-sound');
    if (soundBtn) {
      if (this.sim.soundEnabled) {
        soundBtn.textContent = '🔊 Sound';
        soundBtn.classList.add('active');
      } else {
        soundBtn.textContent = '🔇 Sound';
        soundBtn.classList.remove('active');
      }
    }
  }

  updateStats() {
    document.getElementById('fps-display').textContent = `FPS: ${this.sim.fps}`;
    document.getElementById('count-display').textContent = `Particles: ${this.sim.particles.length}`;
    document.getElementById('zoom-display').textContent = `Zoom: ${this.sim.zoom.toFixed(1)}x`;

    const lifeEl = document.getElementById('life-display');
    if (lifeEl) {
      if (this.sim.lifeEnabled) {
        lifeEl.style.display = '';
        lifeEl.textContent = `Births: ${this.sim.births} · Deaths: ${this.sim.deaths}`;
      } else {
        lifeEl.style.display = 'none';
      }
    }
  }

  /* ---- Saved presets ---- */
  loadSavedPresets() {
    const container = document.getElementById('saved-presets');
    container.innerHTML = '';
    const presets = JSON.parse(localStorage.getItem('pl_presets') || '{}');

    for (const name of Object.keys(presets)) {
      const btn = document.createElement('button');
      btn.textContent = `📂 ${name}`;
      btn.addEventListener('click', () => {
        this.sim.importConfig(presets[name]);
      });
      container.appendChild(btn);
    }
  }
}
