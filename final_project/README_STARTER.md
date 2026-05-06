# Dynamic Travel Replanning — Student Starter Pack

This release gives you everything you need to develop on the **public** benchmark:
- public episodes
- deterministic simulator data
- transparent evaluator
- official runtime wrapper for model usage / cost logging
- a blank submission file and a working baseline example

## What you should edit

Start with these files:
- `student_solver.py` — your submission entrypoint
- `student_solver_example.py` — working baseline example
- `run_student.py` — simplest way to run your solver locally
- `STUDENT_EVALUATION.md` — what gets scored
- `STUDENT_TOOLING.md` — what tools you may use and extend

You do **not** need to edit the simulator data files.

## Recommended first run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.sample .env
# add OPENAI_API_KEY to .env
cp llm_eval_config.json llm_eval_config_student.json
# modify your configs

python run_student.py --solver student_solver_example --limit-public 2 --output-dir runs/example_smoke
```

Then implement `solve_episode(runtime)` in `student_solver.py` and try:

```bash
python run_student.py --solver student_solver --limit-public 3 --output-dir runs/my_solver_smoke
```

## Advanced usage

If you want more control after the basic workflow works, you can still call `run_llm_baselines.py` directly. Most students should start with `run_student.py`.

```bash
python run_llm_baselines.py \
  --config llm_eval_config_student.json \
  --systems student_solver \
  --skip-hidden \
  --skip-ablations \
  --output-dir runs/my_solver_public
```

## Budget knobs

The assignment expects you to tune a bounded search / reasoning budget. Use `--set` to change allowed knobs.

```bash
python run_student.py \
  --solver student_solver \
  --set student_solver.max_tool_rounds=12 \
  --set student_solver.max_output_tokens=1200
```

Hard caps remain enforced by the official config.

## Submission

Submit your `student_solver.py`, `llm_eval_config_student.json` plus any helper Python modules you import from it. Staff will rerun your code with the official runtime wrapper on hidden episodes.
