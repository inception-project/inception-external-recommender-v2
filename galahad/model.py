from typing import Any, Dict, List

from pydantic import BaseModel

Layer = List[Dict[str, Any]]


# Datasets


class DocumentAddRequest(BaseModel):
    name: str  # Name of the document
    text: str  # Document text
    version: int  # Version of the document, needs to be strictly monotonically increasing
    annotations: Dict[
        str, Layer
    ]  # The annotations in the document, one dict per type, start and end offsets index into `text`

    class Config:
        schema_extra = {
            "example": {
                "name": "example_doc",
                "text": "Joe waited for the train . The train was late .",
                "version": 23,
                "annotations": {
                    "g.token": [
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
                    "g.sentence": [
                        {"begin": 0, "end": 26},
                        {"begin": 27, "end": 47},
                    ],
                },
            }
        }


class DocumentListResponse(BaseModel):
    documents: List[str]


# Model

ModelMetaData = Dict[str, Any]


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
                    "token_type": "g.token",
                    "target_type": "g.ner",
                    "target_feature": "label",
                },
                "text": "Joe waited for the train . The train was late .",
                "data": {
                    "g.token": [
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
                    "g.sentence": [
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
                    "g.ner": [
                        {"begin": 0, "end": 3},
                    ]
                }
            }
        }
