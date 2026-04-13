"""학습된 프레이밍 분류 모델을 Hugging Face Hub에 업로드

사용법:
  1. pip install huggingface_hub
  2. huggingface-cli login   (토큰 입력)
  3. python scripts/upload_model.py

옵션:
  --repo   HF 저장소 이름 (기본: klue-roberta-framing-classifier)
  --model  로컬 모델 경로 (기본: models/framing/best)
  --private  비공개 저장소로 생성
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload(repo_name: str, model_dir: str, private: bool = False):
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {model_path}")

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"

    # 저장소 생성 (이미 있으면 무시)
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"저장소: https://huggingface.co/{repo_id}")

    # 업로드
    print(f"업로드 중... ({model_path})")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"\n업로드 완료! https://huggingface.co/{repo_id}")
    print(f"\n사용법:")
    print(f'  from transformers import AutoModelForSequenceClassification, AutoTokenizer')
    print(f'  model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{repo_id}")')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HF Hub 모델 업로드")
    parser.add_argument("--repo", default="klue-roberta-framing-classifier", help="HF 저장소 이름")
    parser.add_argument("--model", default="models/framing/best", help="로컬 모델 경로")
    parser.add_argument("--private", action="store_true", help="비공개 저장소로 생성")
    args = parser.parse_args()

    upload(args.repo, args.model, args.private)
