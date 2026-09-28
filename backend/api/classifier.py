"""학습된 3-class 프레이밍 모델(KLUE-RoBERTa-large)로 실시간 분류

모델은 첫 요청 때 한 번만 로드해 재사용한다 (로드에 수 초, 가중치 1.3GB).
입력 구성은 scripts/train_framing.py 와 동일해야 한다: "제목 [SEP] 본문[:500]", 최대 256 토큰.
"""

import threading
from pathlib import Path

from django.conf import settings

MODEL_DIR = Path(settings.BASE_DIR).parent / "models" / "framing" / "best"
MAX_LENGTH = 256
CONTENT_CHARS = 500

_lock = threading.Lock()
_model = None


class ModelNotAvailable(Exception):
    pass


def _load():
    global _model
    with _lock:
        if _model is None:
            if not (MODEL_DIR / "model.safetensors").exists():
                raise ModelNotAvailable(
                    f"모델 가중치가 없습니다: {MODEL_DIR} (scripts/train_framing.py 로 학습 필요)"
                )
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
            _model = (tokenizer, model, device)
    return _model


def build_text(title: str, content: str = "") -> str:
    title, content = title.strip(), (content or "").strip()
    if len(content) > 10:
        return f"{title} [SEP] {content[:CONTENT_CHARS]}"
    return title


def classify(title: str, content: str = "") -> dict:
    import torch

    tokenizer, model, device = _load()
    enc = tokenizer(build_text(title, content), max_length=MAX_LENGTH, truncation=True,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)[0].cpu().tolist()
    id2label = model.config.id2label
    scores = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
    return {
        "label": max(scores, key=scores.get),
        "probabilities": scores,
        "used_content": "[SEP]" in build_text(title, content),
    }
