# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- Read `spec.md` before non-trivial changes; add an ADR there when you make a design decision.
- Run tests with `.venv/bin/python -m pytest -q tests/test_logic.py` after every change.
- Real pipeline runs take minutes. Start them in the background, write a log, and poll it, e.g.
  `nohup uadub examples/ab/clip.mp4 --voice st -o /tmp/x.mp4 > /tmp/run.log 2>&1 &`.
  Use `--stop-after translate` or a copied `--workdir` to avoid touching the sample outputs.
- Do not overwrite the user's sample results in `examples/` unless asked; write experiments to `/tmp`.
- The user communicates in Ukrainian: reply in Ukrainian, keep code and docs in English (README.uk.md in Ukrainian).
- Do not commit or push unless asked.
