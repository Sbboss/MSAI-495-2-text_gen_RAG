#!/usr/bin/env python3
"""
Run checkpoint inference + quality checks and save JSON results.

Usage example:
python scripts/run_inference_eval.py \
  --checkpoint-path checkpoints/gpt2-so-jax/20260527_120000/latest \
  --output artifacts/inference_eval_latest.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml
from tokenizers import Tokenizer
from tqdm import tqdm

import flax.linen as nn
import orbax.checkpoint as ocp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference eval for JAX GPT2 checkpoint.")
    p.add_argument("--params", type=str, default="params.yml", help="Path to params.yml")
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Checkpoint path (e.g., .../latest or .../step_1234). If empty, resolve from --run-dir + --checkpoint-name.",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Run directory under checkpoints root (e.g., checkpoints/gpt2-so-jax/20260527_120000).",
    )
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default="latest",
        help="Checkpoint folder inside run dir (default: latest).",
    )
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--min-new-tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.35)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.20,
        help="Subtract count * penalty from seen-token logits.",
    )
    p.add_argument(
        "--presence-penalty",
        type=float,
        default=0.05,
        help="Subtract penalty once for any seen token.",
    )
    p.add_argument(
        "--eos-bias",
        type=float,
        default=1.5,
        help="Add this logit bias to EOS after min-new-tokens.",
    )
    p.add_argument("--context-window", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--grounded-format",
        action="store_true",
        help="Wrap prompts with a grounding template (cause/fix/code) before generation.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="artifacts/inference_eval_results.json",
        help="Output JSON file path.",
    )
    return p.parse_args()


def _set_runtime_env(project_root: Path, cfg: dict[str, Any]) -> dict[str, str]:
    runtime_cfg = cfg.get("runtime", {})
    tmp_dir = (project_root / runtime_cfg.get("tmp_dir", "artifacts/jax_tmp")).resolve()
    xdg_dir = (project_root / runtime_cfg.get("xdg_cache_dir", "artifacts/xdg_cache")).resolve()
    cuda_dir = (project_root / runtime_cfg.get("cuda_cache_dir", "artifacts/cuda_cache")).resolve()
    jax_cache_dir = (
        project_root / runtime_cfg.get("jax_compilation_cache_dir", "artifacts/jax_cache")
    ).resolve()
    for p in (tmp_dir, xdg_dir, cuda_dir, jax_cache_dir):
        p.mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["XDG_CACHE_HOME"] = str(xdg_dir)
    os.environ["CUDA_CACHE_PATH"] = str(cuda_dir)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(jax_cache_dir)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    tempfile.tempdir = str(tmp_dir)

    return {
        "TMPDIR": os.environ["TMPDIR"],
        "XDG_CACHE_HOME": os.environ["XDG_CACHE_HOME"],
        "CUDA_CACHE_PATH": os.environ["CUDA_CACHE_PATH"],
        "JAX_COMPILATION_CACHE_DIR": os.environ["JAX_COMPILATION_CACHE_DIR"],
    }


class CausalSelfAttention(nn.Module):
    n_embd: int
    n_head: int
    dropout: float

    @nn.compact
    def __call__(self, x, train: bool):
        bsz, tsz, csz = x.shape
        head_dim = csz // self.n_head
        qkv = nn.Dense(3 * csz, use_bias=True, name="qkv_proj")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(bsz, tsz, self.n_head, head_dim).transpose(0, 2, 1, 3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        att = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(head_dim)
        causal = jnp.tril(jnp.ones((tsz, tsz), dtype=jnp.bool_))[None, None, :, :]
        att = jnp.where(causal, att, -1e10)
        att = nn.softmax(att, axis=-1)
        att = nn.Dropout(rate=self.dropout)(att, deterministic=not train)

        y = jnp.einsum("bhqk,bhkd->bhqd", att, v)
        y = y.transpose(0, 2, 1, 3).reshape(bsz, tsz, csz)
        y = nn.Dense(csz, use_bias=True, name="out_proj")(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=not train)
        return y


class MLP(nn.Module):
    n_embd: int
    dropout: float

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(4 * self.n_embd, name="fc")(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(self.n_embd, name="proj")(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return x


class Block(nn.Module):
    n_embd: int
    n_head: int
    dropout: float

    @nn.compact
    def __call__(self, x, train: bool):
        x = x + CausalSelfAttention(self.n_embd, self.n_head, self.dropout, name="attn")(
            nn.LayerNorm(name="ln_1")(x), train=train
        )
        x = x + MLP(self.n_embd, self.dropout, name="mlp")(nn.LayerNorm(name="ln_2")(x), train=train)
        return x


class GPT2LM(nn.Module):
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    max_seq_len: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, train: bool):
        _, tsz = input_ids.shape
        tok_emb = nn.Embed(self.vocab_size, self.n_embd, name="wte")
        pos_emb = self.param(
            "wpe",
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, self.n_embd),
        )
        x = tok_emb(input_ids) + pos_emb[:tsz][None, :, :]
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        for i in range(self.n_layer):
            x = Block(self.n_embd, self.n_head, self.dropout, name=f"h_{i}")(x, train=train)
        x = nn.LayerNorm(name="ln_f")(x)
        logits = jnp.einsum("btc,vc->btv", x, tok_emb.embedding)
        return logits


@dataclass
class PromptItem:
    category: str
    prompt: str


PROMPTS: list[PromptItem] = [
    PromptItem(
        "so_style",
        "<bos> Question:\nHow do I fix 'TypeError: list object has no attribute strip' in Python when a column can be list or string?\nTags: python pandas\n<sep> Answer:\n",
    ),
    PromptItem(
        "so_style",
        "<bos> Question:\nJAX pmap only sees one GPU although two GPUs are available. What should I check first?\nTags: jax gpu\n<sep> Answer:\n",
    ),
    PromptItem(
        "so_style",
        "<bos> Question:\nDuckDB join on large parquet runs out of memory. What memory-safe strategy should I use?\nTags: duckdb parquet\n<sep> Answer:\n",
    ),
    PromptItem(
        "debugging",
        "<bos> Question:\nI get RESOURCE_EXHAUSTED: /tmp tempfile Disk quota exceeded during JAX training compile. How can I fix it on cluster?\nTags: jax hpc\n<sep> Answer:\n",
    ),
    PromptItem(
        "debugging",
        "<bos> Question:\nFile Save Error in Jupyter notebook says disk I/O error, but df -h shows free space. Why?\nTags: jupyter linux\n<sep> Answer:\n",
    ),
    PromptItem(
        "debugging",
        "<bos> Question:\nTokenization is slower with 62 workers than with 1 worker using tokenizers encode_batch. Why does this happen?\nTags: python multiprocessing\n<sep> Answer:\n",
    ),
    PromptItem(
        "chat_style",
        "<bos> User:\nMy preprocessing pipeline keeps crashing from memory usage. Give me a practical step-by-step fix plan.\nAssistant:\n",
    ),
    PromptItem(
        "chat_style",
        "<bos> User:\nI trained on 20 shards and quality is weak. Should I change model size, data, or decoding first?\nAssistant:\n",
    ),
    PromptItem(
        "chat_style",
        "<bos> User:\nHow do I structure checkpoints for multiple runs so resume is easy and reproducible?\nAssistant:\n",
    ),
    PromptItem(
        "code_help",
        "<bos> Question:\nWrite a Python function that safely converts input that may be str, list, bytes, dict, or None into clean text.\nTags: python\n<sep> Answer:\n",
    ),
    PromptItem(
        "code_help",
        "<bos> Question:\nHow to implement no-repeat-ngram decoding for autoregressive generation in Python pseudocode?\nTags: nlp decoding\n<sep> Answer:\n",
    ),
    PromptItem(
        "code_help",
        "<bos> Question:\nGive a minimal checklist to verify JAX is actually using 2 GPUs with pmap.\nTags: jax cuda\n<sep> Answer:\n",
    ),
]


def _clean_text(text: str) -> str:
    text = text.replace("Ġ", " ").replace("Ċ", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def _ground_prompt(prompt: str) -> str:
    """Add lightweight structure so generations stay tied to user question."""
    if "<sep> Answer:" in prompt:
        q_part, _ = prompt.split("<sep> Answer:", 1)
        q_part = q_part.rstrip()
        return (
            f"{q_part}\n"
            "Answer requirements:\n"
            "1) Root cause in 1-2 lines.\n"
            "2) Concrete fix steps.\n"
            "3) Minimal code/example.\n"
            "4) Keep answer specific to this question only.\n"
            "<sep> Answer:\n"
        )
    # Chat-style fallback.
    return (
        f"{prompt.rstrip()}\n"
        "Response requirements:\n"
        "- Explain root cause first.\n"
        "- Give actionable steps.\n"
        "- Add one concise code/example block when relevant.\n"
        "- Stay specific to the question.\n"
    )


def _resolve_checkpoint_path(project_root: Path, cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.checkpoint_path:
        p = Path(args.checkpoint_path)
        return p if p.is_absolute() else (project_root / p)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir = run_dir if run_dir.is_absolute() else (project_root / run_dir)
        return run_dir / args.checkpoint_name

    runs_root = project_root / cfg["paths"]["checkpoint_dir"]
    run_dirs = sorted([p for p in runs_root.glob("*") if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs found in {runs_root}")
    return run_dirs[-1] / args.checkpoint_name


def _apply_repetition_penalty(logits: jnp.ndarray, generated_ids: list[int], penalty: float) -> jnp.ndarray:
    if penalty <= 1.0 or not generated_ids:
        return logits
    for tok in set(generated_ids):
        val = logits[tok]
        logits = logits.at[tok].set(jnp.where(val > 0, val / penalty, val * penalty))
    return logits


def _apply_frequency_presence_penalty(
    logits: jnp.ndarray,
    generated_ids: list[int],
    *,
    frequency_penalty: float,
    presence_penalty: float,
) -> jnp.ndarray:
    if not generated_ids or (frequency_penalty <= 0 and presence_penalty <= 0):
        return logits
    counts: dict[int, int] = {}
    for tok in generated_ids:
        counts[tok] = counts.get(tok, 0) + 1
    for tok, cnt in counts.items():
        penalty = (frequency_penalty * cnt) + (presence_penalty if cnt > 0 else 0.0)
        if penalty > 0:
            logits = logits.at[tok].add(-penalty)
    return logits


def _ban_no_repeat_ngram_logits(logits: jnp.ndarray, generated_ids: list[int], n: int) -> jnp.ndarray:
    if n <= 1 or len(generated_ids) < n - 1:
        return logits
    prefix = tuple(generated_ids[-(n - 1) :])
    banned = set()
    for i in range(len(generated_ids) - n + 1):
        gram = tuple(generated_ids[i : i + n])
        if gram[:-1] == prefix:
            banned.add(gram[-1])
    if banned:
        idx = jnp.asarray(list(banned), dtype=jnp.int32)
        logits = logits.at[idx].set(-1e10)
    return logits


def sample_next_token(
    logits_last: jnp.ndarray,
    rng_key,
    *,
    generated_ids: list[int],
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    frequency_penalty: float,
    presence_penalty: float,
    eos_id: int | None,
    eos_bias: float,
    cur_new_tokens: int,
    min_new_tokens: int,
):
    logits = logits_last
    logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    logits = _apply_frequency_presence_penalty(
        logits,
        generated_ids,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    logits = _ban_no_repeat_ngram_logits(logits, generated_ids, no_repeat_ngram_size)
    if eos_id is not None:
        if cur_new_tokens < min_new_tokens:
            logits = logits.at[eos_id].set(-1e10)
        elif eos_bias > 0:
            logits = logits.at[eos_id].add(eos_bias)
    if temperature <= 0:
        return int(jnp.argmax(logits)), rng_key

    logits = logits / temperature
    if top_k > 0:
        top_vals, top_idx = jax.lax.top_k(logits, k=min(top_k, logits.shape[-1]))
        probs = jax.nn.softmax(top_vals)
        rng_key, sub = jax.random.split(rng_key)
        choice = jax.random.categorical(sub, jnp.log(probs))
        return int(top_idx[choice]), rng_key

    probs = jax.nn.softmax(logits)
    rng_key, sub = jax.random.split(rng_key)
    return int(jax.random.categorical(sub, jnp.log(probs))), rng_key


def _build_fixed_context(ids: list[int], context_window: int, pad_id: int) -> jnp.ndarray:
    ctx = ids[-context_window:]
    if len(ctx) < context_window:
        ctx = [pad_id] * (context_window - len(ctx)) + ctx
    return jnp.asarray([ctx], dtype=jnp.int32)


def _token_repetition_ratio(ids: list[int], n: int = 3) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return 0.0
    unique = len(set(grams))
    return float(1.0 - (unique / len(grams)))


def _keyword_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower()))


def _auto_scores(prompt: str, output: str, out_ids: list[int]) -> dict[str, Any]:
    prompt_kw = _keyword_set(prompt)
    out_kw = _keyword_set(output)
    overlap = len(prompt_kw & out_kw)
    overlap_ratio = overlap / max(1, len(prompt_kw))
    rep_ratio = _token_repetition_ratio(out_ids, n=3)

    has_steps = bool(re.search(r"(\n- |\n\d+[\).]|first|then|finally|step)", output.lower()))
    has_code = ("```" in output) or bool(re.search(r"\b(def|import|class|return|SELECT|FROM)\b", output))
    long_enough = len(output.split()) >= 40

    task_match = 2 if overlap_ratio >= 0.15 else 1 if overlap_ratio >= 0.08 else 0
    grounding = 2 if overlap_ratio >= 0.20 else 1 if overlap_ratio >= 0.10 else 0
    coherence = 2 if rep_ratio < 0.12 else 1 if rep_ratio < 0.22 else 0
    actionability = 2 if has_steps and long_enough else 1 if has_steps or has_code else 0
    technicality = 2 if has_code and overlap_ratio >= 0.10 else 1 if has_code else 0

    return {
        "task_match_0_2": task_match,
        "grounding_0_2": grounding,
        "coherence_0_2": coherence,
        "actionability_0_2": actionability,
        "technicality_0_2": technicality,
        "auto_total_0_10": task_match + grounding + coherence + actionability + technicality,
        "signals": {
            "prompt_keyword_overlap_ratio": round(overlap_ratio, 4),
            "trigram_repetition_ratio": round(rep_ratio, 4),
            "has_steps": has_steps,
            "has_code_like_tokens": has_code,
            "word_count": len(output.split()),
        },
    }


def _looks_like_template_noise(text: str) -> bool:
    patterns = [
        r"\bi\s*'\s*m using\b",
        r"\bthe following\b",
        r"\bfile file\b",
        r"\bgit\b",
        r"\b\.net\b",
        r"`{3}",
    ]
    s = text.lower()
    hits = sum(1 for pat in patterns if re.search(pat, s))
    return hits >= 2


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    if project_root.name == "scripts":
        project_root = project_root.parent

    params_path = Path(args.params)
    if not params_path.is_absolute():
        params_path = project_root / params_path
    with params_path.open() as f:
        cfg = yaml.safe_load(f)

    runtime_env = _set_runtime_env(project_root, cfg)
    ckpt_path = _resolve_checkpoint_path(project_root, cfg, args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    processed_dir = project_root / cfg["paths"]["processed_dir"]
    tokenizer_path = processed_dir / cfg["paths"]["tokenizer_relpath"]
    hf_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = hf_tokenizer.get_vocab_size()
    pad_token = cfg["data"].get("pad_token", "<pad>")
    pad_id = hf_tokenizer.token_to_id(pad_token)
    eos_id = hf_tokenizer.token_to_id("<eos>")
    if pad_id is None:
        raise ValueError(f"Tokenizer missing pad token id for token: {pad_token}")

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(str(ckpt_path))
    params = restored["params"]
    model_cfg = restored.get("model_config", {})
    n_layer = int(model_cfg.get("n_layer", cfg["model"]["n_layer"]))
    n_head = int(model_cfg.get("n_head", cfg["model"]["n_head"]))
    n_embd = int(model_cfg.get("n_embd", cfg["model"]["n_embd"]))
    max_seq_len = int(model_cfg.get("max_seq_len", cfg["data"]["max_seq_len"]))
    dropout = float(cfg["model"].get("dropout", 0.1))

    model = GPT2LM(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    @jax.jit
    def _next_logits(local_params, input_ids):
        logits = model.apply({"params": local_params}, input_ids, train=False)
        return logits[:, -1, :]

    def encode(text: str) -> list[int]:
        return hf_tokenizer.encode(text).ids

    def decode(ids: list[int]) -> str:
        return _clean_text(hf_tokenizer.decode(ids))

    context_window = min(args.context_window, max_seq_len)
    # Warmup compile with a fixed shape.
    warm_ids = _build_fixed_context([pad_id], context_window, pad_id)
    _ = _next_logits(params, warm_ids)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(tqdm(PROMPTS, desc="Generating prompts")):
        prompt_for_gen = _ground_prompt(item.prompt) if args.grounded_format else item.prompt
        ids = encode(prompt_for_gen)
        gen_ids = list(ids)
        rng_key = jax.random.PRNGKey(args.seed + i)
        prompt_len = len(gen_ids)

        for _ in range(args.max_new_tokens):
            x = _build_fixed_context(gen_ids, context_window, pad_id)
            logits_last = _next_logits(params, x)[0]
            nxt, rng_key = sample_next_token(
                logits_last,
                rng_key,
                generated_ids=gen_ids,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                eos_id=eos_id,
                eos_bias=args.eos_bias,
                cur_new_tokens=len(gen_ids) - prompt_len,
                min_new_tokens=args.min_new_tokens,
            )
            gen_ids.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break

        generated_text = decode(gen_ids)
        answer_text = decode(gen_ids[prompt_len:])
        if "\nQuestion:" in answer_text:
            answer_text = answer_text.split("\nQuestion:", 1)[0].strip()
        if "\nUser:" in answer_text:
            answer_text = answer_text.split("\nUser:", 1)[0].strip()
        if "\nTags:" in answer_text:
            answer_text = answer_text.split("\nTags:", 1)[0].strip()
        scores = _auto_scores(item.prompt, answer_text, gen_ids[prompt_len:])
        noise_flag = _looks_like_template_noise(answer_text)
        results.append(
            {
                "category": item.category,
                "prompt": item.prompt,
                "prompt_used_for_generation": prompt_for_gen,
                "generated_text_full": generated_text,
                "generated_answer_only": answer_text,
                "prompt_tokens": prompt_len,
                "generated_tokens": len(gen_ids) - prompt_len,
                "ended_with_eos": bool(eos_id is not None and gen_ids[-1] == eos_id),
                "template_noise_flag": noise_flag,
                "auto_scores": scores,
                "manual_score_template": {
                    "task_match_0_2": None,
                    "technical_correctness_0_2": None,
                    "actionability_0_2": None,
                    "coherence_0_2": None,
                    "grounding_0_2": None,
                    "notes": "",
                },
            }
        )

    dt = time.time() - t0
    avg_auto = sum(r["auto_scores"]["auto_total_0_10"] for r in results) / max(1, len(results))
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eos_ended = sum(1 for r in results if r.get("ended_with_eos"))
    noise_count = sum(1 for r in results if r.get("template_noise_flag"))
    report = {
        "metadata": {
            "project_root": str(project_root),
            "params_path": str(params_path),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_step": int(restored.get("step", -1)),
            "runtime_env": runtime_env,
            "tempfile_gettempdir": tempfile.gettempdir(),
            "jax_devices": [str(d) for d in jax.devices()],
            "decode_settings": {
                "max_new_tokens": args.max_new_tokens,
                "min_new_tokens": args.min_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
                "eos_bias": args.eos_bias,
                "context_window": context_window,
            },
            "grounded_format": bool(args.grounded_format),
            "model_config": {
                "vocab_size": vocab_size,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "max_seq_len": max_seq_len,
            },
            "elapsed_seconds": round(dt, 2),
            "num_prompts": len(results),
            "avg_auto_total_0_10": round(avg_auto, 4),
            "eos_end_rate": round(eos_ended / max(1, len(results)), 4),
            "template_noise_rate": round(noise_count / max(1, len(results)), 4),
            "created_at_unix": int(time.time()),
        },
        "results": results,
    }

    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved inference report to: {out_path}")
    print(f"Avg auto score (0-10): {avg_auto:.3f} over {len(results)} prompts")


if __name__ == "__main__":
    main()
