const { chromium } = require('playwright');

// Force X11, disable Wayland
process.env.WAYLAND_DISPLAY = '';
process.env.XDG_SESSION_TYPE = 'x11';

(async () => {
  const browser = await chromium.launch({
    headless: false,
    args: [
      '--ozone-platform=x11',
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist'
    ]
  });

  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  
  // Check WebGPU
  const gpuStatus = await page.evaluate(async () => {
    if (!navigator.gpu) return 'WebGPU not in navigator (expected with Xvfb - no GPU)';
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? 'Adapter found: ' + adapter.info?.description : 'Adapter is null';
  });
  console.log('WebGPU:', gpuStatus);

  // Navigate to a page to prove rendering works
  await page.goto('about:blank');
  await page.setContent(`<html><body style="margin:0;display:flex;align-items:center;justify-content:center;height:100vh;background:#1a1a2e;color:white;font-family:sans-serif"><h1>Headed Chrome Works! ✓</h1></body></html>`);
  
  await page.screenshot({ path: 'test-screenshot.png' });
  console.log('Screenshot saved: test-screenshot.png');

  await browser.close();
  console.log('Done!');
  process.exit(0);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
