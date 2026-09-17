# Lab authoring contract

Read `labs/ch01_first_model.py`, `labs/solutions/ch01_first_model.py`, and
`labs/tests/test_ch01.py` first — they are the reference implementation.

## Files per chapter NN

| File | Purpose |
|---|---|
| `labs/chNN_<topic>.py` | Starter the learner edits. Module docstring = the problem brief. Each task is a function with a full docstring (inputs, outputs, an example), a `# TODO:` hint line, and `raise NotImplementedError("Task: <name>")`. Helper/data functions (`make_*_data`) are complete and marked "Do not modify". A `__main__` block prints results so F5 in VS Code shows progress. |
| `labs/solutions/chNN_<topic>.py` | Identical file with every task implemented. Same function names/signatures. |
| `labs/tests/test_chNN.py` | pytest; imports `from labs import chNN_<topic> as lab`; one or two tests per task, named `test_taskK_<what>`; deterministic (seeded) data; assert behavior, shapes, ranges, metric thresholds with margin, and reproducibility. |

## The brief (module docstring)

1. **THE PROBLEM** — a realistic work scenario in 4–8 sentences (a business, a decision, the data you have). Make it concrete: names, numbers, constraints.
2. **TASKS** — numbered list, one line each, mapping to the functions.
3. **STRETCH** (optional) — 1–2 open-ended prompts with no test.

## Task design

- 5–7 tasks per lab, in rising difficulty; the first is a warm-up, the last integrates the chapter (e.g., a full Pipeline + CV + threshold choice).
- Every task exercises something the chapter teaches; name the chapter section in the docstring when helpful.
- Data: generated with `numpy.random.default_rng(seed)` or `sklearn.datasets` offline loaders / `make_*`. Never download. Never write files outside `labs/data/` (create it with `os.makedirs(..., exist_ok=True)` if needed).
- Runtime: the whole test file must finish in < 60 s on CPU (torch labs: tiny nets, few epochs; set `torch.manual_seed`).
- Tests must fail on the starter (NotImplementedError) and pass on the solution. Verify with `python3 tools/check_labs.py chNN`.
- Include at least one test that catches the classic mistake (e.g., leakage: fitting the scaler on the full data; wrong split; using accuracy on imbalanced data).
- Return plain Python/NumPy/pandas objects from tasks (floats, dicts, arrays, DataFrames, fitted sklearn objects) — easy to assert.
- Keep starters readable in VS Code: type hints, short functions, no clever tricks.
