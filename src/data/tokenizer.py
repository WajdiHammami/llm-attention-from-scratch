import sentencepiece as spm


tokenizer = None

def get_tokenizer(path: str = "src/data/processed/tokenizer.model"):
    global tokenizer
    if tokenizer is None:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(path)
    return tokenizer


def encode(text: str, tokenizer: spm.SentencePieceProcessor):
    return tokenizer.encode(text, out_type=int)  # return list of ids


def decode(ids: list[int], tokenizer: spm.SentencePieceProcessor):
    return tokenizer.decode(ids)  # return string


def vocab_size(tokenizer: spm.SentencePieceProcessor):
    return tokenizer.get_piece_size()
