import os

from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
SUDOKU_CHARS = set("0123456789")


def build_tokenizer():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_list = SPECIAL_TOKENS + sorted(SUDOKU_CHARS)
    vocab_dict = {ch: i for i, ch in enumerate(vocab_list)}

    print(f"Sudoku characters: {sorted(SUDOKU_CHARS)}")
    print(f"Vocabulary size (including specials): {len(vocab_dict)}")

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab_dict, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        mask_token="[MASK]",
    )

    os.makedirs(out_dir, exist_ok=True)
    hf_tokenizer.save_pretrained(out_dir)
    print(f"\nSaved sudoku character tokenizer to {out_dir}/")

    test_str = "530070000"
    print("\nSanity check:")
    print(f"Input:    '{test_str}'")
    print(f"Tokens:   {hf_tokenizer.tokenize(test_str)}")
    print(f"Token IDs:{hf_tokenizer.encode(test_str)}")


if __name__ == "__main__":
    build_tokenizer()
