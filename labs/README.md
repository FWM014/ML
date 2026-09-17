# Labs — real problems to solve in VS Code

One problem set per chapter. Each lab is a Python file with `TODO` blocks; a matching
pytest file turns green as you solve them. This is how you actually learn: read the
chapter, then make the tests pass.

## Setup (once)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Open this folder in VS Code (`File → Open Folder`). Accept the recommended extensions
(Python, Pylance, Jupyter, Live Preview). Select the `.venv` interpreter
(`Ctrl/Cmd+Shift+P → Python: Select Interpreter`).

## Workflow

1. Read the chapter in the book (`index.html` — right-click → *Show Preview* with the
   Live Preview extension, or open it in your browser).
2. Open `labs/chNN_<topic>.py`. Read the docstring at the top: it states the business
   problem, the data, and the tasks.
3. Fill in each `TODO`. Run the file (F5 → "Run current lab file") to see your own
   prints, or run the tests: the **Testing** sidebar (beaker icon) → `labs/tests/test_chNN.py`.
   Red → green, task by task.
4. Stuck for more than 20 minutes? Peek at `labs/solutions/chNN_<topic>.py`, understand
   it, close it, and write your own version.

Run everything from the command line:

```bash
pytest labs/tests -q                 # all labs
pytest labs/tests/test_ch05.py -q    # one chapter
```

## Layout

```
labs/chNN_<topic>.py          starter with TODOs   ← you edit this
labs/tests/test_chNN.py       tests for the starter
labs/solutions/chNN_<topic>.py reference solution (same function names)
labs/data/                    small generated datasets (created by the labs themselves)
```

## Rules of the game

- Every lab runs offline in under a minute on a laptop CPU.
- Tests check behavior (shapes, ranges, metric thresholds), not exact numbers, unless a
  value is deterministic.
- Function signatures in the starter are the contract: keep them; the tests import them.
