import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig


@dataclass
class TextClassificationModelData:
    model_name: str
    label: str
    score: float


class BaseTextClassificationModel(ABC):

    def __init__(self, name: str, model_path: str, tokenizer: str):
        self.name = name
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._load_model()

    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def __call__(self, inputs) -> List[TextClassificationModelData]:
        ...


class TransformerTextClassificationModel(BaseTextClassificationModel):
    def __init__(self, name: str, model_path: str, tokenizer: str):
        super().__init__(name, model_path, tokenizer)
        self.optimized_dir = "optimized_onnx_models"
        os.makedirs(self.optimized_dir, exist_ok=True)
        self.optimized_model_path = os.path.join(self.optimized_dir, f"{self.name}_optimized.onnx")

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if not os.path.exists(self.optimized_model_path):
            onnx_path = self._export_to_onnx()
            self._optimize_onnx(onnx_path)
        self._load_model()

    def _export_to_onnx(self) -> str:
        model_pt = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        model_pt.to(self.device)
        model_pt.eval()

        dummy_text = "Test sentence."
        inputs = self.tokenizer(dummy_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        raw_onnx_path = os.path.join(self.optimized_dir, f"{self.name}.onnx")

        torch.onnx.export(model_pt, tuple(inputs.values()), raw_onnx_path, opset_version=14, export_params=True,
                          input_names=list(inputs.keys()), output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "attention_mask": {0: "batch_size"},
                                        "output": {0: "batch_size"}})
        return raw_onnx_path

    def _optimize_onnx(self, raw_onnx_path: str):
        optimization_config = OptimizationConfig(optimization_level=2, optimize_for_gpu=(self.device == 0))
        optimizer = ORTOptimizer.from_pretrained(model=self.name, file_name=os.path.basename(raw_onnx_path),
                                                 framework="transformers")
        optimizer.optimize(save_dir=self.optimized_dir, optimization_config=optimization_config)

    def _load_model(self):
        self.model = ORTModelForSequenceClassification.from_pretrained(
            self.optimized_dir,
            file_name=f"{self.name}_optimized.onnx",
            provider="CUDAExecutionProvider" if self.device == 0 else "CPUExecutionProvider"
        )

    def tokenize_texts(self, texts: List[str]):
        inputs = self.tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                return_token_type_ids=True,
                return_tensors='pt'
                )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to GPU
        return inputs

    def __call__(self, inputs) -> List[TextClassificationModelData]:
        logits = self.model(**inputs).logits
        label_ids = logits.argmax(dim=1)
        scores = logits.softmax(dim=-1)
        id2label = self.model.config.id2label

        predictions = []
        for i in range(label_ids.size(0)):
            label_id = label_ids[i].item()
            score = scores[i, label_id].item()
            predictions.append(TextClassificationModelData(
                model_name=self.name,
                label=id2label[label_id],
                score=score
            ))
        return predictions

