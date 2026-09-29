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
            conv: chart('#chConv'), power: chart('#chPower'), soc: chart('#chSoc'), price: chart('#chPrice')};
  }, tag);
  const setSlider = (id, v) => page.evaluate((id, v) => {
    const el = document.querySelector('#' + id); el.value = v;
    el.dispatchEvent(new Event('input')); el.dispatchEvent(new Event('change'));
  }, id, v);

  const steps = [];
  await idle();
  await setSlider('max_iter', 20);                   // keep the ADMM solves short
  await idle();
  steps.push(await snap('dw'));
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
