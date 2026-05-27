#!/usr/bin/env python3
"""Train GPT-2 (JAX/Flax) from params.yml in terminal/tmux/screen."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from tokenizers import Tokenizer
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT-2 with JAX/Flax using params.yml.")
    p.add_argument("--params", type=str, default="params.yml", help="Path to params.yml")
    p.add_argument("--num-steps", type=int, default=0, help="Override training.max_steps if > 0")
    p.add_argument("--eval-batches", type=int, default=64, help="Validation batches at eval step")
    p.add_argument("--seed", type=int, default=-1, help="Override training.seed if >= 0")
    return p.parse_args()


def _setup_runtime_env(project_root: Path, cfg: dict) -> None:
    runtime_cfg = cfg.get("runtime", {})
    tmp_dir = (project_root / runtime_cfg.get("tmp_dir", "artifacts/jax_tmp")).resolve()
    xdg_dir = (project_root / runtime_cfg.get("xdg_cache_dir", "artifacts/xdg_cache")).resolve()
    cuda_dir = (project_root / runtime_cfg.get("cuda_cache_dir", "artifacts/cuda_cache")).resolve()
    jax_cache_dir = (
        project_root / runtime_cfg.get("jax_compilation_cache_dir", "artifacts/jax_cache")
    ).resolve()
    for p in (tmp_dir, xdg_dir, cuda_dir, jax_cache_dir):
        p.mkdir(parents=True, exist_ok=True)

    for var in ("TMPDIR", "TEMP", "TMP"):
        os.environ[var] = str(tmp_dir)
    os.environ["XDG_CACHE_HOME"] = str(xdg_dir)
    os.environ["CUDA_CACHE_PATH"] = str(cuda_dir)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(jax_cache_dir)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    tempfile.tempdir = str(tmp_dir)


def _list_array_to_numpy(list_array, seq_len: int) -> np.ndarray:
    """Convert Arrow ListArray[int] -> writable np.ndarray[B, T]."""
    try:
        vals = np.asarray(list_array.values.to_numpy(), dtype=np.int32)
        return np.array(vals.reshape(len(list_array), seq_len), dtype=np.int32, copy=True)
    except Exception:
        return np.array(list_array.to_pylist(), dtype=np.int32, copy=True)


def iter_parquet_batches(
    chunk_paths: list[str],
    batch_size: int,
    *,
    seq_len: int,
    row_batch_multiplier: int,
    repeat: bool,
    shuffle_files: bool,
    shuffle_rows: bool,
    drop_last: bool,
    seed: int,
):
    rng = np.random.default_rng(seed)
    paths = list(chunk_paths)
    while True:
        if shuffle_files:
            rng.shuffle(paths)
        for path in paths:
            pf = pq.ParquetFile(path)
            row_batch_size = max(batch_size * row_batch_multiplier, 256)
            for rb in pf.iter_batches(batch_size=row_batch_size, columns=["input_ids"]):
                arr = _list_array_to_numpy(rb.column(0), seq_len)
                if shuffle_rows:
                    rng.shuffle(arr)
                for i in range(0, len(arr), batch_size):
                    b = arr[i : i + batch_size]
                    if len(b) < batch_size and drop_last:
                        continue
                    yield b
        if not repeat:
            break


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

    _setup_runtime_env(project_root, cfg)

    # Import JAX/Flax after runtime env setup.
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from flax import jax_utils
    from flax.training import train_state

    processed_dir = project_root / cfg["paths"]["processed_dir"]
    tokenizer_path = processed_dir / cfg["paths"]["tokenizer_relpath"]
    train_glob = str(processed_dir / cfg["paths"]["train_glob"])
    val_glob = str(processed_dir / cfg["paths"]["val_glob"])
    runs_root = project_root / cfg["paths"]["checkpoint_dir"]
    metrics_filename = cfg["paths"].get("metrics_filename", "metrics.json")

    max_seq_len = int(cfg["data"]["max_seq_len"])
    pad_token = cfg["data"].get("pad_token", "<pad>")
    row_batch_multiplier = int(cfg.get("dataloader", {}).get("row_batch_multiplier", 4))

    n_layer = int(cfg["model"]["n_layer"])
    n_head = int(cfg["model"]["n_head"])
    n_embd = int(cfg["model"]["n_embd"])
    dropout = float(cfg["model"]["dropout"])

    global_batch_size = int(cfg["training"]["global_batch_size"])
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"]["weight_decay"])
    max_steps = int(cfg["training"]["max_steps"])
    warmup_steps = int(cfg["training"]["warmup_steps"])
    log_every = int(cfg["training"].get("log_every", 20))
    eval_every = int(cfg["training"]["eval_every"])
    save_every = int(cfg["training"]["save_every"])
    seed = int(cfg["training"]["seed"])
    if args.num_steps > 0:
        max_steps = args.num_steps
    if args.seed >= 0:
        seed = args.seed

    resume_training = bool(cfg["resume"].get("enabled", False))
    resume_checkpoint = cfg["resume"].get("checkpoint", "latest")
    resume_run_dir = cfg["resume"].get("run_dir", "")

    use_pmap = bool(cfg.get("distributed", {}).get("use_pmap", True))
    require_num_devices = int(cfg.get("distributed", {}).get("require_num_devices", 1))
    num_devices = jax.local_device_count()
    if use_pmap and num_devices < require_num_devices:
        raise RuntimeError(
            f"USE_PMAP=True but only {num_devices} device(s) visible. Expected >= {require_num_devices}."
        )
    if use_pmap and (global_batch_size % num_devices != 0):
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by device_count ({num_devices})"
        )
    per_device_batch_size = global_batch_size // num_devices if use_pmap else global_batch_size

    if resume_training:
        if not resume_run_dir:
            raise ValueError("resume.run_dir must be set when resume.enabled=true")
        run_dir = project_root / resume_run_dir
    else:
        run_dir = runs_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = run_dir
    metrics_path = run_dir / metrics_filename
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "params_snapshot.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_chunks = sorted(glob.glob(train_glob))
    val_chunks = sorted(glob.glob(val_glob))
    if not train_chunks:
        raise FileNotFoundError(f"No training chunks found: {train_glob}")
    if not val_chunks:
        raise FileNotFoundError(f"No validation chunks found: {val_glob}")

    hf_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = hf_tokenizer.get_vocab_size()
    pad_id = hf_tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise ValueError(f"pad token id not found for token: {pad_token}")

    print("Project:", project_root)
    print("Params:", params_path)
    print("Run dir:", run_dir)
    print("TMPDIR:", os.environ.get("TMPDIR"))
    print("JAX devices:", jax.devices())
    print(f"USE_PMAP={use_pmap} NUM_DEVICES={num_devices} GLOBAL_BATCH_SIZE={global_batch_size}")

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

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, drop_rng = jax.random.split(rng, 3)
    model = GPT2LM(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    dummy_ids = jnp.ones((1, max_seq_len), dtype=jnp.int32)
    variables = model.init({"params": init_rng, "dropout": drop_rng}, dummy_ids, train=True)
    params = variables["params"]
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model params: {n_params/1e6:.2f}M")

    def make_batch(np_ids: np.ndarray) -> dict[str, jnp.ndarray]:
        ids = jnp.asarray(np_ids, dtype=jnp.int32)
        return {"input_ids": ids, "labels": ids}

    def shard_batch(batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        if not use_pmap:
            return batch

        def _reshape(x):
            x = np.asarray(x)
            new_shape = (num_devices, per_device_batch_size) + x.shape[1:]
            return jnp.asarray(x.reshape(new_shape), dtype=jnp.int32)

        return {k: _reshape(v) for k, v in batch.items()}

    def causal_lm_loss(logits: jnp.ndarray, labels: jnp.ndarray, pad_id_local: int = pad_id) -> jnp.ndarray:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = (shift_labels != pad_id_local).astype(jnp.float32)
        token_loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
        token_loss = token_loss * mask
        return token_loss.sum() / jnp.maximum(mask.sum(), 1.0)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=max_steps,
        end_value=lr * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, b1=0.9, b2=0.95),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    if use_pmap:
        state = jax_utils.replicate(state)

    checkpointer = ocp.PyTreeCheckpointer()

    def _state_to_host(st):
        return jax_utils.unreplicate(st) if use_pmap else st

    def save_ckpt(path: Path, st, rng_key, step: int):
        host_state = _state_to_host(st)
        payload = {
            "params": jax.device_get(host_state.params),
            "opt_state": jax.device_get(host_state.opt_state),
            "step": int(step),
            "rng": jax.device_get(rng_key),
            "model_config": {
                "vocab_size": vocab_size,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "max_seq_len": max_seq_len,
            },
        }
        if path.exists():
            shutil.rmtree(path)
        checkpointer.save(str(path), payload)
        print("saved", path)

    def restore_ckpt(path: Path, st):
        restored = checkpointer.restore(str(path))
        host_state = _state_to_host(st)
        host_state = host_state.replace(params=restored["params"], opt_state=restored["opt_state"])
        out_state = jax_utils.replicate(host_state) if use_pmap else host_state
        step = int(restored["step"])
        rng_key = restored["rng"]
        print(f"restored {path} @ step {step}")
        return out_state, rng_key, step

    @jax.jit
    def train_step_single(st, batch, dropout_rng):
        def loss_fn(local_params):
            logits = st.apply_fn({"params": local_params}, batch["input_ids"], train=True, rngs={"dropout": dropout_rng})
            return causal_lm_loss(logits, batch["labels"], pad_id)

        loss, grads = jax.value_and_grad(loss_fn)(st.params)
        st = st.apply_gradients(grads=grads)
        return st, loss

    @partial(jax.pmap, axis_name="devices")
    def train_step_pmap(st, batch, dropout_rng):
        def loss_fn(local_params):
            logits = st.apply_fn({"params": local_params}, batch["input_ids"], train=True, rngs={"dropout": dropout_rng})
            return causal_lm_loss(logits, batch["labels"], pad_id)

        loss, grads = jax.value_and_grad(loss_fn)(st.params)
        loss = jax.lax.pmean(loss, axis_name="devices")
        grads = jax.lax.pmean(grads, axis_name="devices")
        st = st.apply_gradients(grads=grads)
        return st, loss

    @jax.jit
    def eval_step_single(local_params, batch):
        logits = model.apply({"params": local_params}, batch["input_ids"], train=False)
        return causal_lm_loss(logits, batch["labels"], pad_id)

    @partial(jax.pmap, axis_name="devices")
    def eval_step_pmap(local_params, batch):
        logits = model.apply({"params": local_params}, batch["input_ids"], train=False)
        loss = causal_lm_loss(logits, batch["labels"], pad_id)
        return jax.lax.pmean(loss, axis_name="devices")

    def append_metric(record: dict):
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text())
        else:
            data = []
        data.append(record)
        metrics_path.write_text(json.dumps(data, indent=2))

    def evaluate_perplexity(st, max_batches: int) -> tuple[float, float]:
        losses = []
        val_iter = iter_parquet_batches(
            val_chunks,
            batch_size=global_batch_size,
            seq_len=max_seq_len,
            row_batch_multiplier=row_batch_multiplier,
            repeat=False,
            shuffle_files=False,
            shuffle_rows=False,
            drop_last=False,
            seed=seed,
        )
        for i, batch_np in enumerate(val_iter):
            if i >= max_batches:
                break
            batch = make_batch(batch_np)
            if use_pmap:
                batch = shard_batch(batch)
                loss = eval_step_pmap(st.params, batch)
                losses.append(float(jax.device_get(loss)[0]))
            else:
                loss = eval_step_single(st.params, batch)
                losses.append(float(loss))
        if not losses:
            return 0.0, 1.0
        mean_loss = float(np.mean(losses))
        return mean_loss, float(np.exp(mean_loss))

    start_step = 0
    if resume_training:
        ckpt_path = checkpoint_dir / resume_checkpoint
        if ckpt_path.exists():
            state, rng, start_step = restore_ckpt(ckpt_path, state)
        else:
            print(f"[warn] Resume requested but checkpoint does not exist: {ckpt_path}")

    train_iter = iter_parquet_batches(
        train_chunks,
        batch_size=global_batch_size,
        seq_len=max_seq_len,
        row_batch_multiplier=row_batch_multiplier,
        repeat=True,
        shuffle_files=True,
        shuffle_rows=True,
        drop_last=True,
        seed=seed,
    )
    running = []
    pbar = tqdm(range(start_step + 1, max_steps + 1), desc="Training", unit="step", dynamic_ncols=True)
    for step in pbar:
        batch_np = next(train_iter)
        batch = make_batch(batch_np)

        if use_pmap:
            batch = shard_batch(batch)
            if step == start_step + 1:
                pbar.write(f"pmap batch shape: {batch['input_ids'].shape} (num_devices, per_device_batch, seq)")
            rng, base = jax.random.split(rng)
            step_rng = jax.random.split(base, num_devices)
            state, loss = train_step_pmap(state, batch, step_rng)
            loss_scalar = float(jax.device_get(loss)[0])
            state_step = int(jax.device_get(state.step)[0])
        else:
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step_single(state, batch, step_rng)
            loss_scalar = float(loss)
            state_step = int(state.step)

        running.append(loss_scalar)

        if step % log_every == 0:
            mean_loss = float(np.mean(running[-log_every:]))
            lr_now = float(lr_schedule(state_step))
            train_ppl = float(np.exp(mean_loss))
            pbar.set_postfix(loss=f"{mean_loss:.4f}", ppl=f"{train_ppl:.2f}", lr=f"{lr_now:.2e}")
            append_metric({"step": step, "train_loss": mean_loss, "train_ppl": train_ppl, "lr": lr_now})

        if step % eval_every == 0:
            val_loss, val_ppl = evaluate_perplexity(state, max_batches=args.eval_batches)
            pbar.write(f"[eval] step={step:6d} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
            append_metric({"step": step, "val_loss": val_loss, "val_ppl": val_ppl})

        if step % save_every == 0:
            save_ckpt(checkpoint_dir / f"step_{step}", state, rng, step)
            save_ckpt(checkpoint_dir / "latest", state, rng, step)

    final_step = int(jax.device_get(state.step)[0]) if use_pmap else int(state.step)
    save_ckpt(checkpoint_dir / "latest", state, rng, final_step)
    print(f"Training complete. Final step: {final_step}")
    print(f"Run dir: {run_dir}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
