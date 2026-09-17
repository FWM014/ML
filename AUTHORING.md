# Authoring guide — *Machine Learning: Zero to Expert*

This file is the contract every chapter follows. Read it fully, then read
`chapters/01-what-is-ml.html` as the reference implementation, before writing a chapter.

## 0. Ground rules

1. **Every chapter is a single self-contained HTML file** in `chapters/`, named exactly as in `assets/chapters.js`.
2. **Never edit** `assets/style.css`, `assets/app.js`, or `assets/chapters.js`. If you need a style, use inline `style=""` sparingly or an SVG attribute.
3. **No external resources.** No CDN scripts, no images, no web fonts. Figures are inline SVG. Math is HTML (`<sub>`, `<sup>`, Unicode: × · ≈ ≤ ≥ ∑ ∂ √ σ μ λ α β θ ∈ →) inside `<div class="formula">`.
4. **Voice:** a senior data scientist who is also a patient professor. Direct, concrete, opinionated about what matters at work. No filler. Every section teaches something the reader can *use*.
5. **Audience:** a smart professional who took one intro course, wants to apply ML at work fast, and needs to become expert. Assume nothing is remembered; explain hard ideas "like I'm 9" first, then precisely.
6. **Correctness beats coverage.** Every formula, every number, every claim must be right. Every Python block must run.

## 1. File skeleton (copy exactly)

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NN · Chapter Title — ML: Zero to Expert</title>
<link rel="stylesheet" href="../assets/style.css">
</head>
<body data-chapter="NN">
<header class="topbar"></header>
<div class="progress-track"><div></div></div>
<div class="shell">
<nav class="sidebar"></nav>
<main class="content"><div class="container">

<header class="ch-header">
  <div class="kicker">Part X · Part Name · Chapter NN</div>
  <h1>Chapter Title</h1>
  <p class="lede">One or two sentences: why this chapter matters and what you'll be able to do.</p>
  <div class="meta"><span class="level intermediate">Intermediate</span> <span>⏱ 60 min read</span> <span>·</span> <span>Prereqs: Ch 5, 6</span></div>
</header>

<div class="objectives"><h3>By the end of this chapter you will be able to</h3><ul>…4–6 items…</ul></div>
<div class="toc-inline"><strong>In this chapter</strong><ol>…links to each h2 id…</ol></div>

… sections (h2 id="s1" … ), figures, callouts, code, widgets …

<h2 id="quiz">Quiz</h2> … <h2 id="exercises">Exercises</h2> … <div class="summary">…</div>

<div class="mark-done"></div>
<nav class="pager"></nav>

</div></main>
</div>
<script src="../assets/chapters.js"></script>
<script src="../assets/app.js"></script>
<script> /* widget code, wrapped in (function(){ … })(); */ </script>
</body>
</html>
```

`data-chapter="NN"` (two digits) is what makes the sidebar, pager and progress work. Level pill classes: `level` (beginner, default), `level intermediate`, `level advanced`, `level expert`.

## 2. Components

| Component | Markup |
|---|---|
| ELI9 box | `<div class="callout eli9"><div class="title">Explain it like I'm 9</div><p>…</p></div>` |
| Key idea | `<div class="callout key"><div class="title">Key idea</div>…</div>` |
| Common mistake | `<div class="callout warn"><div class="title">Common mistake</div>…</div>` |
| At work / pro tip | `<div class="callout pro"><div class="title">At work</div>…</div>` |
| Optional math | `<div class="callout math"><div class="title">The math (optional)</div>…</div>` |
| Formula | `<div class="formula">ŷ = w<sub>0</sub> + w<sub>1</sub>x<sub>1</sub> + … + w<sub>n</sub>x<sub>n</sub></div>` |
| Figure | `<figure class="fig"><svg viewBox="0 0 W H" role="img" aria-label="…">…</svg><figcaption><strong>Figure N.k — Title.</strong> One or two sentences stating the claim the picture makes.</figcaption></figure>` |
| Code | `<pre class="code" data-lang="python"><code>…</code></pre>` with manual highlighting spans: `.cm` comment, `.kw` keyword, `.fn` function/builtin, `.st` string, `.nu` number, `.out` expected output line (write output as `# …` comment lines with class `out`). **HTML-escape `<`, `>`, `&` inside code.** |
| Table | `<div class="tbl-wrap"><table class="tbl"><thead>…</thead><tbody>…</tbody></table></div>` (`td.num` right-aligns) |
| Cards | `<div class="grid2">` or `grid3` containing `<div class="card [good|bad|accent]"><h4>…</h4><p>…</p></div>` |
| Numbered steps | `<ol class="steps"><li><strong>Title.</strong> body…</li></ol>` |
| Widget | `<div class="widget" id="w-x"><div class="w-title">Interactive: …</div><div class="controls"><label>Name <input type="range" id="x" min max value data-out="x-out"> <output id="x-out"></output></label> <span>Readout: <span class="readout" id="x-r">–</span></span></div><svg viewBox="…" id="x-svg" role="img" aria-label="…"></svg><div class="note">What to notice.</div></div>` — `data-out` auto-mirrors the range value into the `<output>`. |
| Quiz | `<div class="quiz"><h3>Check your understanding</h3><div class="q"><p class="question">1. …?</p><ul class="opts"><li><button>A</button></li><li><button data-correct>B</button></li>…</ul><div class="explain">Why B.</div></div>…</div>` — exactly one `data-correct` per question. |
| Exercise | `<div class="exercise"><div class="title">Exercise N.k — Name</div><p>…</p><details class="answer"><summary>Show answer</summary>…</details></div>` |
| Summary | `<div class="summary"><h3>Chapter summary</h3><ul class="checklist"><li>…</li></ul></div>` |
| Legend | `<div class="legend"><span style="--c:var(--s1)">Series A</span><span style="--c:var(--s2)">Series B</span></div>` |
| Deep dive (collapsible) | `<details class="deepdive"><summary>Deep dive: Title <span class="time">8 min</span></summary> …any content: p, figures, code, math boxes… </details>` — violet accent, 🔬 icon, collapsed by default. `app.js` gives each one an id (`dd-1`, `dd-2`, …) and builds a "Deep dives in this chapter" box (`.deepdive-index`) right after `.toc-inline`, so keep the summary text short and start it with "Deep dive:". |
| Deep dive heading (non-collapsible) | `<h3 class="deepdive-h">Title</h3>` — for the rare always-visible in-depth subsection; same violet accent bar. |
| Lab | `<div class="callout lab"><div class="title">Lab</div>…</div>` — 🧪 hands-on task box (teal tint, distinct from `pro`). |

## 3. SVG figure rules

- Hand-authored inline SVG with native shapes and `<text>`. Set `viewBox`; never width/height attributes.
- **Colors:** series via `fill="var(--s1)"` … `var(--s8)` (or classes `fill-s1`, `stroke-s1`); ink via `currentColor`, `var(--ink)`, `var(--ink-2)`, `var(--ink-3)`; surfaces via `var(--surface)`, `var(--surface-2)`, `var(--accent-soft)`; status via `var(--good)`, `var(--critical)`, `var(--warn)`. Both light and dark themes must read well — never hardcode black/white/gray hex.
- Helper classes inside `figure.fig svg`: `.axis`, `.grid`, `.lbl` (12px secondary text), `.lbl-strong` (bold), `.box` (neutral node), `.box-accent` (highlighted node), `.arrow` (stroke, add your own `marker-end`).
- Arrowheads: `<defs><marker id="UNIQUE" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>` — **marker ids must be unique per page** (e.g. `arr-f3`).
- Use the series colors in fixed order: first series `--s1` (blue), second `--s2` (orange), third `--s3` (aqua), fourth `--s4`. Decision boundaries / highlights: `--s7` (violet). Errors/bad: `var(--critical)`. Good: `var(--good)`.
- Text 11–14px at drawn scale, `text-anchor` for alignment. Label the axes. Label the arrows. Keep at least 20px padding so nothing clips; **check the right edge**.
- One figure = one claim, stated in the caption. Number figures `Figure N.k` sequentially within the chapter.
- Prefer diagrams that show the *mechanism* (how data flows, what a split does, what a gradient step does) over decorative icons. Draw real data shapes (scatter, curves, bars, heatmap cells, trees, network layers) with plausible coordinates.
- No `<style>`, `<script>`, `<foreignObject>`, or images inside SVG.

## 4. Widgets (interactive figures)

Each chapter includes **1–2 interactive widgets** where the concept benefits (a slider that moves a parameter and redraws, a step button that advances an algorithm, a toggle that adds/removes regularization). Vanilla JS, rendered into an `<svg>` by setting `innerHTML`, wrapped in an IIFE, no globals, no external libs. Keep each under ~120 lines. Use `var(--s1)` etc. in generated markup. Must not throw: test in the browser (see §7).

## 5. Chapter content requirements

- **Length:** 3,500–6,000 words of prose (excluding code). 6–10 `h2` sections plus Quiz, Exercises, Summary.
- **Figures:** 6–10 `figure.fig` SVGs + 1–2 widgets.
- **Callouts:** ≥ 4 `eli9`, ≥ 3 `key`, ≥ 2 `warn`, ≥ 3 `pro`, and `math` boxes wherever a formula appears (formula shown, then each symbol explained in words).
- **Code:** ≥ 4 Python blocks. Use scikit-learn / pandas / numpy (PyTorch for Part III). Blocks should be complete and runnable on a synthetic or built-in dataset (`sklearn.datasets`, `make_classification`, `make_regression`, `load_breast_cancer`, `fetch_california_housing` is NOT allowed—network—use `load_diabetes`, `load_iris`, `load_wine`, or generated data). Include expected output as `.out` comment lines. Show the sklearn *pattern* (`fit`/`predict`/`Pipeline`) the reader will use at work.
- **Quiz:** 6 questions, 4 options each, one correct, each with an `explain`.
- **Exercises:** 3, with answers in `details.answer` (at least one is a "do it at work" exercise).
- **Summary:** 6–10 checklist lines.
- **Cross-references:** point to other chapters by number ("see Chapter 13") — see `assets/chapters.js` for the map.
- Every technical term gets an ELI9-style analogy the first time it appears; then the precise definition.
- Include at least one **"which one do I use at work?"** decision table or flow figure where the chapter covers multiple methods.

## 6. Code verification

Run `python3 tools/check_code.py chapters/NN-file.html`. It extracts every `python` block, unescapes it, and executes each block in a fresh process; it reports tracebacks. All blocks must pass (blocks that intentionally show an error should be marked `data-lang="python-noexec"`). Installed: numpy, pandas, scikit-learn, matplotlib, xgboost, lightgbm, statsmodels, shap, torch (CPU).

## 7. Visual verification

Run `node tools/shot.js chapters/NN-file.html /tmp/NN.png` (and again with a third arg `dark`). It prints page height and any JS errors — **errors must be `[]`**. Open the PNG (crop into ~1800px slices) and look for clipped SVG text, overlapping labels, empty figures, or broken widgets. Fix and re-shoot until clean.

## 8. Checklist before you're done

- [ ] `data-chapter` correct; `<title>` correct; kicker/part correct.
- [ ] Objectives, inline TOC (links match `h2` ids), Quiz (6), Exercises (3), Summary present.
- [ ] Figure count, widget count, callout counts meet §5.
- [ ] All SVG text inside the viewBox in both themes; unique marker ids.
- [ ] `check_code.py` passes; `shot.js` shows `errors: []` in light and dark.
- [ ] No external URLs loaded; no `assets/` edits.
