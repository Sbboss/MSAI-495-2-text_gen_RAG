"""Shared text helpers for StackOverflow preprocessing."""

from __future__ import annotations

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]


def is_val(question_id, val_pct: float) -> bool:
    try:
        qid = int(question_id)
    except (TypeError, ValueError):
        return False
    return (qid * 2654435761) % 10_000 < int(val_pct * 10_000)


def to_text(value, max_chars: int | None = None) -> str:
    if value is None:
        s = ""
    elif isinstance(value, str):
        s = value
    elif isinstance(value, bytes):
        s = value.decode("utf-8", errors="replace")
    elif isinstance(value, (list, tuple)):
        s = ", ".join(p for p in (to_text(v) for v in value) if p)
    elif isinstance(value, dict):
        s = ", ".join(f"{k}: {to_text(v)}" for k, v in value.items() if to_text(v))
    else:
        s = str(value)
    s = s.strip()
    if max_chars is not None and len(s) > max_chars:
        s = s[:max_chars]
    return s


def format_qa_row(
    title,
    question_body,
    tags,
    answer_body,
    max_chars: int,
    prompt_style: str = "standard",
) -> str:
    title = to_text(title, max_chars // 4)
    question_body = to_text(question_body, max_chars // 2)
    tags = to_text(tags, max_chars // 8)
    answer_body = to_text(answer_body, max_chars)
    if prompt_style == "grounded_qa":
        parts = [
            "<bos> Task: Answer the technical question accurately and concisely.",
            "Question:",
        ]
        if title:
            parts.append(f"Title: {title}")
        if question_body:
            parts.append(f"Body: {question_body}")
        if tags:
            parts.append(f"Tags: {tags}")
        parts.extend(
            [
                "Answer requirements:",
                "1) State likely root cause briefly.",
                "2) Provide concrete fix steps.",
                "3) Add a minimal example when relevant.",
                "<sep> Final Answer:",
                answer_body,
                "<eos>",
            ]
        )
        return "\n".join(parts)

    parts = ["<bos> Question:"]
    if title:
        parts.append(title)
    if question_body:
        parts.append(question_body)
    if tags:
        parts.append(f"Tags: {tags}")
    parts.append("<sep> Answer:")
    parts.append(answer_body)
    parts.append("<eos>")
    return "\n".join(parts)


def train_tokenizer(sample_path, out_dir, vocab_size: int) -> Tokenizer:
    def line_iter():
        with sample_path.open(encoding="utf-8") as fh:
            for line in fh:
                yield line.strip()

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(line_iter(), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    return tokenizer


def pad_ids(ids: list[int], max_len: int, pad_id: int) -> list[int]:
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [pad_id] * (max_len - len(ids))
    return ids
