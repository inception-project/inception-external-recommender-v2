from pathlib import Path

from fastapi import FastAPI

from galahad.model import ModelStore
# This regex forbids two consecutive dots so that ../foo does not work
# to discovery files outside of the document folder
from galahad.routes import register_routes


class GalahadServer(FastAPI):
    def __init__(self, title: str = "Galahad Server", data_dir: Path = None) -> None:
        super().__init__(title=title)

        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[1]

        self.state.data_dir = data_dir
        self.state.model_store = ModelStore()

        register_routes(self)
