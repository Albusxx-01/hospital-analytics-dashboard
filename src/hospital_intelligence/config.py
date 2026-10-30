"""Explicit configuration shared by the pipeline and presentation layer."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HOSPITALS = ("Harbor Medical", "Northstar General", "Cedar Valley")
DEPARTMENTS = ("General Medicine", "Cardiology", "Orthopedics", "Pulmonology")
DIAGNOSES = {
    "General Medicine": ("Metabolic disorder", "Infection", "Gastrointestinal condition"),
    "Cardiology": ("Heart failure", "Arrhythmia", "Coronary artery disease"),
    "Orthopedics": ("Fracture", "Joint replacement", "Spinal condition"),
    "Pulmonology": ("Pneumonia", "COPD", "Asthma"),
}


@dataclass(frozen=True)
class Settings:
    root: Path = Path("data")
    seed: int = 42
    encounters: int = 50_000
    start: str = "2024-01-01"
    end: str = "2025-12-31"

    @property
    def cutoff(self) -> pd.Timestamp:
        """Exclusive end of the observation window."""
        return pd.Timestamp(self.end) + pd.Timedelta(days=1)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"


@dataclass(frozen=True)
class Filters:
    start: str
    end: str
    hospital: str = "All hospitals"
    department: str = "All departments"
    age: str = "All ages"
    sex: str = "All sexes"
    diagnosis: str = "All diagnoses"
