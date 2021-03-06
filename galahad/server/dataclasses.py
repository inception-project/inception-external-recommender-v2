from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Annotation(BaseModel):
    begin: int
    end: int
    features: Dict[str, Any] = Field(default_factory=dict)


Layer = List[Annotation]
Layers = Dict[str, Layer]


# Datasets


class DatasetList(BaseModel):
    names: List[str]

    class Config:
        schema_extra = {"example": {"names": ["dataset1", "dataset2", "dataset3"]}}


class Document(BaseModel):
    text: str  # Document text
    annotations: Dict[
        str, Layer
    ]  # The annotations in the document, one dict per type, start and end offsets index into `text`
    version: int = Field(default=0)  # Version of the document, needs to be monotonically increasing

    class Config:
        schema_extra = {
            "example": {
                "text": "Joe waited for the train . The train was late .",
                "version": 23,
                "annotations": {
                    "t.token": [
                        {"begin": 0, "end": 3},
                        {"begin": 4, "end": 10},
                        {"begin": 11, "end": 14},
                        {"begin": 15, "end": 18},
                        {"begin": 19, "end": 24},
                        {"begin": 25, "end": 26},
                        {"begin": 27, "end": 30},
                        {"begin": 31, "end": 36},
                        {"begin": 37, "end": 40},
                        {"begin": 41, "end": 45},
                        {"begin": 46, "end": 47},
                    ],
                    "t.sentence": [
                        {"begin": 0, "end": 26},
                        {"begin": 27, "end": 47},
                    ],
                    "t.named_entity": [
                        {"begin": 0, "end": 3, "features": {"f.value": "PER"}},
                    ],
                },
            }
        }


class DocumentList(BaseModel):
    names: List[str]
    versions: List[int]

    class Config:
        schema_extra = {
            "example": {"names": ["document1.xmi", "document2.txt", "document3.pdf"], "versions": [7, 6, 7]}
        }


# Classifier


class ClassifierInfo(BaseModel):
    name: str

    class Config:
        schema_extra = {"example": {"name": "ExampleClassifier"}}


# Training


class TrainingRequest(BaseModel):
    metadata: Dict[str, Any]


# Predicting


class PredictionRequest(BaseModel):
    metadata: Dict[str, Any]
    text: str
    data: Dict[str, Layer]

    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "token_type": "t.token",
                    "target_type": "t.ner",
                    "target_feature": "label",
                },
                "text": "Joe waited for the train . The train was late .",
                "data": {
                    "t.token": [
                        {"begin": 0, "end": 3},
                        {"begin": 4, "end": 10},
                        {"begin": 11, "end": 14},
                        {"begin": 15, "end": 18},
                        {"begin": 19, "end": 24},
                        {"begin": 25, "end": 26},
                        {"begin": 27, "end": 30},
                        {"begin": 31, "end": 36},
                        {"begin": 37, "end": 40},
                        {"begin": 41, "end": 45},
                        {"begin": 46, "end": 47},
                    ],
                    "t.sentence": [
                        {"begin": 0, "end": 26},
                        {"begin": 27, "end": 47},
                    ],
                },
            }
        }


class PredictionResponse(BaseModel):
    data: Dict[str, Layer]

    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "t.ner": [
                        {"begin": 0, "end": 3},
                    ]
                }
            }
        }
