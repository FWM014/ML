#!/usr/bin/env python3
"""Audit every chapter against AUTHORING.md: structure, counts, ids, links, external resources.
Usage: python3 tools/audit.py [chapters/NN-file.html ...]   (default: all chapters)
"""
import re, sys, glob, os, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "chapters", "*.html")))
manifest = open(os.path.join(ROOT, "assets", "chapters.js"), encoding="utf-8").read()
known = dict(re.findall(r'num:\s*"(\d\d)",\s*file:\s*"([^"]+)"', manifest))

def count(pat, s, flags=re.S): return len(re.findall(pat, s, flags))
def words(s):
    body = re.sub(r"<pre class=\"code\".*?</pre>", " ", s, flags=re.S)
    body = re.sub(r"<script.*?</script>", " ", body, flags=re.S)
    body = re.sub(r"<svg.*?</svg>", " ", body, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    return len(html.unescape(body).split())

problems_total = 0
print(f"{'file':34} {'words':>6} {'fig':>4} {'wid':>4} {'eli9':>5} {'key':>4} {'warn':>5} {'pro':>4} {'math':>5} {'code':>5} {'quiz':>5} {'exer':>5}  issues")
for f in files:
    s = open(f, encoding="utf-8").read()
    name = os.path.basename(f)
    issues = []
    num = re.search(r'data-chapter="(\d\d)"', s)
    num = num.group(1) if num else None
    if not num: issues.append("no data-chapter")
    elif known.get(num) != name: issues.append(f"data-chapter {num} != manifest ({known.get(num)})")
    n = dict(
        fig=count(r'<figure class="fig">', s), wid=count(r'<div class="widget"', s),
        eli9=count(r'class="callout eli9"', s), key=count(r'class="callout key"', s),
        warn=count(r'class="callout warn"', s), pro=count(r'class="callout pro"', s),
        math=count(r'class="callout math"', s), code=count(r'<pre class="code" data-lang="python', s),
        quiz=count(r'<div class="q">', s), exer=count(r'<div class="exercise">', s),
    )
    w = words(s)
    if name != "24-cheatsheets.html":
        if w < 3000: issues.append(f"prose short ({w})")
        if n["fig"] < 6: issues.append("figures<6")
        if n["eli9"] < 4: issues.append("eli9<4")
        if n["key"] < 3: issues.append("key<3")
        if n["warn"] < 2: issues.append("warn<2")
        if n["pro"] < 3: issues.append("pro<3")
        if n["code"] < 4: issues.append("code<4")
    if n["wid"] < 1: issues.append("no widget")
    if n["quiz"] != 6: issues.append(f"quiz={n['quiz']}")
    if n["exer"] < 3: issues.append(f"exercises={n['exer']}")
    # quiz correctness: exactly one data-correct per q
    for i, q in enumerate(re.findall(r'<div class="q">(.*?)<div class="explain">', s, flags=re.S), 1):
        c = q.count("data-correct")
        if c != 1: issues.append(f"q{i} has {c} correct")
    # required structural bits
    for req in ['class="objectives"', 'class="toc-inline"', 'class="summary"', 'class="mark-done"', 'class="pager"', '../assets/app.js', '../assets/chapters.js', '../assets/style.css']:
        if req not in s: issues.append(f"missing {req}")
    # TOC links resolve
    ids = set(re.findall(r'\sid="([^"]+)"', s))
    for href in re.findall(r'<div class="toc-inline">.*?</div>', s, flags=re.S)[:1]:
        for a in re.findall(r'href="#([^"]+)"', href):
            if a not in ids: issues.append(f"toc #{a} missing")
    # duplicate ids
    allids = re.findall(r'\sid="([^"]+)"', s)
    dups = sorted({i for i in allids if allids.count(i) > 1})
    if dups: issues.append("dup ids: " + ",".join(dups[:5]))
    # external resources
    if re.search(r'(src|href)="https?://', s): issues.append("external URL in src/href")
    if re.search(r'<img\b', s): issues.append("<img> used")
    # svg text sanity: hardcoded grays/blacks
    if re.search(r'(fill|stroke)="#(000|333|666|999|ccc|ddd|eee)"', s, re.I): issues.append("hardcoded gray/black hex in svg")
    # figure numbering
    figs = re.findall(r'Figure (\d+)\.(\d+)', s)
    if figs and num and any(int(a) != int(num) for a, b in figs): issues.append("figure number != chapter")
    # cross-chapter links
    for href in re.findall(r'href="(\d\d-[^"#]+\.html)', s):
        if not os.path.exists(os.path.join(ROOT, "chapters", href)): issues.append(f"broken link {href}")
    problems_total += len(issues)
    print(f"{name:34} {w:6d} {n['fig']:4d} {n['wid']:4d} {n['eli9']:5d} {n['key']:4d} {n['warn']:5d} {n['pro']:4d} {n['math']:5d} {n['code']:5d} {n['quiz']:5d} {n['exer']:5d}  {'; '.join(issues) if issues else 'OK'}")
print(f"\n{len(files)} files, {problems_total} issues")
sys.exit(1 if problems_total else 0)
