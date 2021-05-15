from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path(__file__).resolve().parents[1]
