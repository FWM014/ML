// usage: node shot.js <html-path> <out-png> [dark]
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
(async () => {
  const [,, file, out, dark] = process.argv;
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: dark ? 'dark' : 'light' });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); });
  await page.goto('file://' + require('path').resolve(file));
  await page.waitForTimeout(400);
  await page.screenshot({ path: out, fullPage: true });
  const h = await page.evaluate(() => document.documentElement.scrollHeight);
  console.log(JSON.stringify({ file, height: h, errors }));
  await browser.close();
})();
