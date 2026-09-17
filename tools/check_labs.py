#!/usr/bin/env python3
"""Verify labs: (1) the starter imports and its tests FAIL (or at least don't all pass) ,
(2) the solution, swapped in for the starter, makes every test PASS.
Usage: python3 tools/check_labs.py [ch05 ch06 ...]   (default: all chapters with a test file)
"""
import glob, os, re, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABS = os.path.join(ROOT, "labs")
tests = sorted(glob.glob(os.path.join(LABS, "tests", "test_ch*.py")))
want = set(sys.argv[1:])
fails = 0
for t in tests:
    ch = re.search(r"test_(ch\d\d)\.py", t).group(1)
    if want and ch not in want: continue
    starters = glob.glob(os.path.join(LABS, f"{ch}_*.py"))
    if len(starters) != 1: print(f"{ch}: expected exactly one starter labs/{ch}_*.py, found {starters}"); fails += 1; continue
    starter = starters[0]; sol = os.path.join(LABS, "solutions", os.path.basename(starter))
    if not os.path.exists(sol): print(f"{ch}: missing solution {sol}"); fails += 1; continue
    env = dict(os.environ, PYTHONPATH=ROOT, MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    # 1) starter: must import; tests should not all pass
    r = subprocess.run([sys.executable, "-m", "pytest", t, "-q", "-x", "--no-header", "-p", "no:cacheprovider"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=600)
    starter_all_pass = r.returncode == 0
    if "Error" in r.stdout and "collected 0 items" in r.stdout: print(f"{ch}: tests failed to collect\n{r.stdout[-1500:]}"); fails += 1; continue
    # 2) solution swapped in
    backup = tempfile.mkdtemp(); shutil.copy(starter, backup)
    try:
        shutil.copy(sol, starter)
        r2 = subprocess.run([sys.executable, "-m", "pytest", t, "-q", "--no-header", "-p", "no:cacheprovider"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=900)
    finally:
        shutil.copy(os.path.join(backup, os.path.basename(starter)), starter); shutil.rmtree(backup)
    summary = (r2.stdout.strip().splitlines() or [""])[-1]
    ok = r2.returncode == 0
    print(f"{ch}: solution -> {'PASS' if ok else 'FAIL'} ({summary}); starter {'already passes ALL tests (TODOs too easy?)' if starter_all_pass else 'fails as expected'}")
    if not ok: print(r2.stdout[-3000:]); fails += 1
    if starter_all_pass: fails += 1
print(f"\n{'ALL LABS OK' if not fails else str(fails) + ' problem(s)'}")
sys.exit(1 if fails else 0)
