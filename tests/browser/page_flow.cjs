// Drive the testbed page as a user does and report what the charts did.
//
//   node tests/browser/page_flow.cjs <page url>       (run by tests/test_page_browser.py)
//
// Needs puppeteer: PUPPETEER_MODULE names its path, else require('puppeteer').
// Prints one JSON object: a snapshot after each step - the method, the status
// line, and a hash of each chart's pixels - and every page error.
const puppeteer = require(process.env.PUPPETEER_MODULE || 'puppeteer');

(async () => {
  const browser = await puppeteer.launch({headless: 'new', args: ['--no-sandbox'], protocolTimeout: 1800000});
  const page = await browser.newPage();
  await page.setViewport({width: 1400, height: 1000});
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.argv[2], {timeout: 120000});
  const idle = async () => {
    await new Promise(r => setTimeout(r, 1500));
    await page.waitForFunction(() => document.querySelector('#solve').textContent === 'Solve'
      && /\d/.test(document.querySelector('#kPlan').textContent), {timeout: 900000, polling: 300});
  };
  const snap = tag => page.evaluate(tag => {
    const hash = s => { let x = 0; for (let i = 0; i < s.length; i += 61) x = (x * 31 + s.charCodeAt(i)) >>> 0; return x; };
    const chart = id => hash(document.querySelector(id + ' canvas').toDataURL());
    const pt = document.querySelector('#pausedTag');
    return {tag, method: DATA.summary.method, iterations: DATA.summary.iterations,
            status: document.querySelector('#status').textContent.replace(/\s+/g, ' ').slice(0, 120),
            pausedTag: pt && getComputedStyle(pt).display !== 'none' ? pt.textContent : '',
            button: document.querySelector('#solve').textContent,
            conv: chart('#chConv'), power: chart('#chPower'), soc: chart('#chSoc'), price: chart('#chPrice'),
            room: chart('#chRoom'), band: document.querySelector('#bandState').textContent,
            low7: DATA.comfort.room_low ? DATA.comfort.room_low[Math.round(7 / DATA.dt)] : null};
  }, tag);
  const setSlider = (id, v) => page.evaluate((id, v) => {
    const el = document.querySelector('#' + id); el.value = v;
    el.dispatchEvent(new Event('input')); el.dispatchEvent(new Event('change'));
  }, id, v);

  // Drag one dot of the room's comfort band on the Room chart: find
  // it by moving up hour h's column until the cursor offers a drag (the
  // lowest dot there is the band's floor), then pull it up.
  const dragBand = async (h, dy) => {
    await page.evaluate(() => document.querySelector('#chRoom').scrollIntoView());
    const c = await page.evaluate(h => {
      const r = document.querySelector('#chRoom canvas').getBoundingClientRect(), n = DATA.steps;
      const i = Math.min(Math.round(h / DATA.dt), n - 1);
      return {x: r.left + 46 + i / (n - 1) * (r.width - 46 - 12), top: r.top, bottom: r.bottom};   // chart()'s margins
    }, h);
    for (let y = c.bottom - 21; y > c.top + 8; y -= 2) {
      await page.mouse.move(c.x, y);
      if (await page.evaluate(() => document.querySelector('#chRoom canvas').style.cursor) === 'ns-resize') {
        await page.mouse.down(); await page.mouse.move(c.x, y - dy, {steps: 4}); await page.mouse.up();
        return true;
      }
    }
    return false;
  };

  const steps = [];
  await idle();
  await setSlider('max_iter', 20);                   // keep the ADMM solves short
  await idle();
  steps.push(await snap('dw'));
  if (!await dragBand(7, 30)) throw new Error('no comfort-band dot found at 07:00');
  await idle();
  steps.push(await snap('band dragged'));
  await page.evaluate(() => document.querySelector('#bandReset').click());   // it sits in the closed Advanced panel
  await idle();
  steps.push(await snap('band reset'));
  await page.select('#method', 'admm');
  await idle();
  steps.push(await snap('admm'));
  const rows = await page.$$('#iters tr');
  await rows[Math.min(3, rows.length - 1)].click();
  await new Promise(r => setTimeout(r, 1000));
  steps.push(await snap('admm, an iteration clicked'));
  await page.select('#batt_step', 'lp');
  await idle();
  steps.push(await snap('admm, LP batteries'));
  await page.select('#warm_start', 'warm');
  await idle();
  await setSlider('solar_peak', 6);
  await idle();
  steps.push(await snap('admm, warm start'));
  await page.select('#method', 'dw');
  await idle();
  steps.push(await snap('dw again'));

  // pause and resume: a 60-iteration ADMM solve, paused at iteration 8 or later
  await page.select('#method', 'admm');
  await idle();
  steps.push(await snap('admm before pause'));
  await setSlider('max_iter', 60);                   // starts a new solve
  await page.waitForFunction(() => { const m = document.querySelector('#solve').textContent.match(/Solving \((\d+)\//);
    return m && +m[1] >= 8; }, {timeout: 900000, polling: 100});
  await page.click('#solve');                        // pause
  await page.waitForFunction(() => document.querySelector('#solve').textContent === 'Resume', {timeout: 900000, polling: 200});
  await new Promise(r => setTimeout(r, 500));
  steps.push(await snap('paused'));
  await page.click('#solve');                        // resume
  await page.waitForFunction(() => /Solving/.test(document.querySelector('#solve').textContent), {timeout: 60000, polling: 50});
  steps.push(await snap('resumed, running'));
  await idle();
  steps.push(await snap('resumed, done'));
  console.log(JSON.stringify({steps, errors}));
  await browser.close();
})().catch(e => { console.log(JSON.stringify({fatal: e.message})); process.exit(1); });
