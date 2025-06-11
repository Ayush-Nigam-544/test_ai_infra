from __future__ import annotations

import bentoml
from typing import List


bento_image = bentoml.images.Image(python_version="3.11") \
    .python_packages("torch", "transformers")


@bentoml.service(
    image=bento_image,
    resources={"cpu": "4"}
)
class Summarization:
    model_path = bentoml.models.HuggingFaceModel("sshleifer/distilbart-cnn-12-6")

    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline('summarization', model=self.model_path, device=device)

    @bentoml.api(batchable=True)
    def summarize(self, texts: List[str]) -> List[str]:
        results = self.pipeline(texts)
        return [item['summary_text'] for item in results]