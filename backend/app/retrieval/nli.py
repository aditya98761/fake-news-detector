"""Optional stance detection (does an article support / contradict the claim?) via a multilingual NLI model.
Default: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT, multilingual NLI trained on XNLI + a
machine-translated 26-language set, so quality in Hindi is lower than in English). Set NLI_MODEL= to disable. Headlines are weak premises, so results are shown
as stance + confidence, and the verdict logic demands several independent sources."""
import logging
from functools import lru_cache
from ..schemas.schemas import Stance


class StanceModel:
    def __init__(self, name: str):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=name, top_k=None)

    def stance(self, claim: str, evidence: str):
        out = self.pipe({"text": evidence[:600], "text_pair": claim}, truncation=True)
        rows = out[0] if out and isinstance(out[0], list) else out
        scores = {r["label"].lower(): r["score"] for r in rows}
        ent = next((v for k, v in scores.items() if "entail" in k), 0.0)
        con = next((v for k, v in scores.items() if "contradict" in k), 0.0)
        if ent >= 0.6 and ent > con:
            return Stance.supports, float(ent)
        if con >= 0.6 and con > ent:
            return Stance.contradicts, float(con)
        return Stance.related, float(max(ent, con))


log = logging.getLogger("fnd.nli")


@lru_cache(maxsize=3)
def get_stance_model(name):
    """Returns None (and caches it until restart) if the model is disabled or cannot be loaded."""
    if not name:
        return None
    try:
        return StanceModel(name)
    except Exception:
        log.exception("NLI model unavailable; stance detection disabled")
        return None
