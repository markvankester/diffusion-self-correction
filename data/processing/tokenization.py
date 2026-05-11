"""
Tokenization utilities for the MDLM training pipeline used for arithmetic data.
"""


def _tokenize(examples, tokenizer, text_field, insert_eos, add_special_tokens):
    """Shared tokenization: encode text and optionally append EOS."""
    ids = tokenizer(examples[text_field], add_special_tokens=add_special_tokens)["input_ids"]

    if insert_eos:
        eos_id = tokenizer.eos_token_id
        assert eos_id is not None, "Tokenizer must have an eos_token_id defined."
        ids = [seq + ([] if (seq and seq[-1] == eos_id) else [eos_id]) for seq in ids]

    return ids

def tokenize_and_pad(
    examples,
    tokenizer,
    text_field: str = "text",
    seq_length: int = 16,
    insert_eos: bool = False,
    add_special_tokens: bool = False,
    mask_until_token: str | None = None,
):
    """
    Tokenize text examples, append EOS if requested, and truncate to seq_length.

    Padding is deferred to the trainer's DataCollator.
    If mask_until_token is provided, labels up to and including that token are set to -100.
    """
    ids = _tokenize(examples, tokenizer, text_field, insert_eos, add_special_tokens)

    processed_ids = []
    labels = []

    delimiter_id = tokenizer.convert_tokens_to_ids(mask_until_token) if mask_until_token is not None else None

    for seq in ids:
        if len(seq) > seq_length:
            seq = seq[:seq_length]

        seq_lbl = seq[:]
        if delimiter_id is not None and delimiter_id in seq_lbl:
            idx = seq_lbl.index(delimiter_id)
            for i in range(idx + 1):
                seq_lbl[i] = -100

        processed_ids.append(seq)
        labels.append(seq_lbl)

    return {
        "input_ids": processed_ids,
        "labels": labels,
    }

