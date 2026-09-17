#!/usr/bin/env python3
"""Extract every <pre class="code" data-lang="python"> block from a chapter and execute it.
Usage: python3 tools/check_code.py chapters/05-linear-regression.html [--keep]
"""
import html, re, subprocess, sys, tempfile, os, textwrap

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
blocks = re.findall(r'<pre class="code" data-lang="python"[^>]*>(.*?)</pre>', src, flags=re.S)
if not blocks:
    print("no python blocks found"); sys.exit(0)
fails = 0
for i, b in enumerate(blocks, 1):
    code = re.sub(r"<[^>]+>", "", b)          # strip highlighting spans
    code = html.unescape(code)
    code = re.sub(r"^Copy\n", "", code)
    with tempfile.NamedTemporaryFile("w", suffix=f"_block{i}.py", delete=False, dir=tempfile.gettempdir()) as f:
        f.write(code); tmp = f.name
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore", OMP_NUM_THREADS="2")
    try:
        r = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=300, env=env, cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired:
        print(f"[block {i}] TIMEOUT (>300s)"); fails += 1; continue
    if r.returncode != 0:
        fails += 1
        print(f"[block {i}] FAILED\n{'-'*60}\n{code[:1500]}\n{'-'*60}\n{r.stderr[-2500:]}\n")
    else:
        print(f"[block {i}] ok  ({len(code.splitlines())} lines)" + (f"\n{textwrap.indent(r.stdout.strip()[:600], '    ')}" if r.stdout.strip() else ""))
    if "--keep" not in sys.argv:
        os.unlink(tmp)
print(f"\n{len(blocks)} blocks, {fails} failed")
sys.exit(1 if fails else 0)
