"""Translate-then-classify for Hindi/Marathi: the (English) classifier reads a machine translation of the text.
Default model: facebook/nllb-200-distilled-600M (supports hin_Deva, mar_Deva -> eng_Latn).
Caveats from its model card: CC-BY-NC-4.0 (non-commercial), described as a research model, trained on inputs up to 512 tokens,
not intended for document translation. We therefore translate only the first TRANSLATE_MAX_CHARS characters, in short chunks.
Loaded with AutoModelForSeq2SeqLM directly (the 'translation' pipeline was removed in transformers v5)."""
import logging
import re
from functools import lru_cache
from typing import Optional

log = logging.getLogger("fnd.translate")
LANG_CODE = {"hi": "hin_Deva", "mr": "mar_Deva"}
_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def split_chunks(text: str, max_chars: int = 300) -> list:
    """Pack whole sentences into chunks of at most max_chars (a single longer sentence is cut)."""
    chunks, cur = [], ""
    for sent in _SPLIT.split(text.strip()):
        while len(sent) > max_chars:
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(sent[:max_chars]); sent = sent[max_chars:]
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur); cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


class Translator:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
        self.eng = self.tok.convert_tokens_to_ids("eng_Latn")

    def to_english(self, text: str, lang: str) -> str:
        self.tok.src_lang = LANG_CODE[lang]
        out = []
        chunks = split_chunks(text)
        for i in range(0, len(chunks), 4):
            enc = self.tok(chunks[i:i + 4], return_tensors="pt", padding=True, truncation=True, max_length=256)
            with self._torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=self.eng, max_new_tokens=256, num_beams=2)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return " ".join(out).strip()


@lru_cache(maxsize=1)
def get_translator(name: Optional[str]):
    if not name:
        return None
    try:
        return Translator(name)
    except Exception:
        log.exception("translation model unavailable; Hindi/Marathi stay evidence-only")
        return None


def translate_to_english(name: Optional[str], lang: str, text: str) -> Optional[str]:
    t = get_translator(name)
    if not t or lang not in LANG_CODE:
        return None
    try:
        return t.to_english(text, lang) or None
    except Exception:
        log.exception("translation failed")
        return None
