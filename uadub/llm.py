"""LLM backends: mlx-lm in-process (default), a local Ollama server, or an agent CLI
(Claude Code, opencode, Codex, Gemini) used as a plain text model."""

from __future__ import annotations

import json
import re
import urllib.request


class LLM:
    def chat(self, system: str, user: str, *, max_tokens: int = 4096, temperature: float = 0.3) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MLXLLM(LLM):
    def __init__(self, model_id: str):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)

    def chat(self, system: str, user: str, *, max_tokens: int = 4096, temperature: float = 0.3) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        sampler = make_sampler(temp=temperature, top_p=0.95)
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=sampler, verbose=False)

    def close(self) -> None:
        import gc

        import mlx.core as mx

        del self.model, self.tokenizer
        gc.collect()
        mx.clear_cache()


class OllamaLLM(LLM):
    """Talks to a local Ollama server (http://127.0.0.1:11434) — still fully offline."""

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434"):
        self.model, self.host = model, host.rstrip("/")

    def _post(self, path: str, payload: dict, timeout: float = 1800) -> dict:
        req = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def chat(self, system: str, user: str, *, max_tokens: int = 4096, temperature: float = 0.3) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 16384},
        }
        return self._post("/api/chat", payload)["message"]["content"]

    def close(self) -> None:
        try:  # unload the model so TTS gets the memory back
            self._post("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=30)
        except Exception:
            pass


AGENT_LLMS = ("claude", "opencode", "codex", "gemini")


def is_agent_llm(spec: str) -> bool:
    return spec.partition(":")[0] in AGENT_LLMS


def is_local_mlx(spec: str) -> bool:
    return not (spec.startswith("ollama:") or is_agent_llm(spec))


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class AgentCLILLM(LLM):
    """An installed agent CLI as a text model: «claude:sonnet», «opencode:anthropic/claude-sonnet-4-5»,
    «codex:gpt-5», «gemini:gemini-2.5-pro» (model optional). Runs headless in an empty temp folder,
    without tools, and returns the reply text. Uses the CLI's own login/subscription; NOT offline."""

    def __init__(self, spec: str, timeout: float = 900):
        import shutil

        self.harness, _, self.model = spec.partition(":")
        if not shutil.which(self.harness):
            raise RuntimeError(f"Не знайдено команду «{self.harness}» — встановіть її або оберіть інший --llm")
        self.timeout = timeout

    def _run(self, cmd: list[str], stdin: str | None, cwd: str) -> str:
        import os
        import subprocess

        # Translation needs no deliberation; thinking made each Claude call 3–6× slower (70–140 s vs 23 s).
        env = {**os.environ, "MAX_THINKING_TOKENS": os.environ.get("UADUB_AGENT_THINKING", "0")}
        r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, cwd=cwd, timeout=self.timeout, env=env)
        if r.returncode != 0 and not r.stdout.strip():
            err = (r.stderr or "") + (r.stdout or "")
            msg = re.findall(r'"message"\s*:\s*"([^"]+)"', err) or re.findall(r"(?:Error|ERROR)[:\s]+(.{20,300})", err)
            raise RuntimeError(f"{self.harness} завершився з помилкою: {(msg[-1] if msg else err.strip()[-500:])}")
        return _ANSI.sub("", r.stdout)

    def chat(self, system: str, user: str, *, max_tokens: int = 4096, temperature: float = 0.3) -> str:
        import tempfile
        from pathlib import Path

        m = self.model
        both = f"{system}\n\n---\nINPUT:\n{user}\n\nReply with the JSON only, no other text, do not use tools."
        with tempfile.TemporaryDirectory(prefix="uadub-llm-") as tmp:
            if self.harness == "claude":
                cmd = ["claude", "-p", "--system-prompt", system, "--tools", "", "--output-format", "text",
                       "--no-session-persistence"] + (["--model", m] if m else [])
                return self._run(cmd, user, tmp)
            if self.harness == "opencode":
                return self._run(["opencode", "run"] + (["-m", m] if m else []) + [both], None, tmp)
            if self.harness == "codex":
                out = Path(tmp) / "reply.txt"
                cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
                       "-c", "model_reasoning_effort=low",
                       "-o", str(out)] + (["-m", m] if m else []) + ["-"]
                text = self._run(cmd, both, tmp)
                return out.read_text() if out.exists() else text
            if self.harness == "gemini":
                cmd = ["gemini", "--skip-trust", "-o", "text", "--approval-mode", "plan"] + (["-m", m] if m else []) + ["-p", both]
                return self._run(cmd, None, tmp)
        raise ValueError(self.harness)


def make_llm(spec: str) -> LLM:
    if spec.startswith("ollama:"):
        return OllamaLLM(spec.split(":", 1)[1])
    if is_agent_llm(spec):
        return AgentCLILLM(spec)
    return MLXLLM(spec)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>|<\|channel\|>.*?<\|message\|>", re.S)


def extract_json(text: str):
    """Parse the first JSON object/array in an LLM reply (tolerates fences and chatter)."""
    text = _THINK.sub("", text)
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        a, b = text.find(open_ch), text.rfind(close_ch)
        if a != -1 and b > a:
            try:
                return json.loads(text[a : b + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"LLM не повернула коректний JSON: {text[:300]!r}")
