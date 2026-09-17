// usage: node shot.js <html-path> <out-png> [dark|light]   (no 3rd arg = the page default, which is dark)
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
(async () => {
  const [,, file, out, mode] = process.argv;
  const dark = mode === 'dark';
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: dark ? 'dark' : 'light' });
  const page = await ctx.newPage();
  if (mode === 'light') await page.addInitScript(() => { try { localStorage.setItem('ml-theme', 'light'); } catch (e) {} });
  if (mode === 'dark') await page.addInitScript(() => { try { localStorage.setItem('ml-theme', 'dark'); } catch (e) {} });
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
