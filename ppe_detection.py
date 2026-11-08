"""PPE model download, detection normalization, and alert validation."""

from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


MODEL_REPOSITORY = "killuminati1/construction-ppe-yolov8"
MODEL_REVISION = "06ce70134a78f31881e2f9a223719d60a0a9a6a9"
MODEL_SHA256 = "31ef3ca04a17cf545f3fcfc64c4af8993a41d52ccc460e82aff01d5354603533"
MODEL_FILENAME = "construction-ppe-yolov8.pt"
MODEL_URL = (
    "https://huggingface.co/"
    f"{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/best.pt?download=true"
)

ISSUE_BY_CLASS = {
    "nohelmet": "helmet_missing",
    "nohardhat": "helmet_missing",
    "novest": "vest_missing",
    "nosafetyvest": "vest_missing",
    "noboot": "shoes_missing",
    "noboots": "shoes_missing",
    "nosafetyshoe": "shoes_missing",
    "nosafetyshoes": "shoes_missing",
}

ISSUE_LABELS = {
    "helmet_missing": "Helmet missing (model alert)",
    "vest_missing": "Safety vest missing (model alert)",
    "shoes_missing": "Boots / safety footwear missing (model alert)",
}


def normalize_class_name(name: str) -> str:
    """Normalize model class labels for stable comparisons."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def issue_for_class(label: str) -> str | None:
    """Map a supported missing-PPE model class to an incident type."""
    return ISSUE_BY_CLASS.get(normalize_class_name(label))


def download_ppe_model(destination: Path) -> Path:
    """Download a pinned public checkpoint and verify its SHA-256 digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    digest = hashlib.sha256()

    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            with temporary_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise RuntimeError(
                "The downloaded PPE model failed SHA-256 verification. "
                "The incomplete file was not installed."
            )
        temporary_path.replace(destination)
        return destination
    except (OSError, urllib.error.URLError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the PPE model: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class PPEDetection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def issue_type(self) -> str | None:
        return issue_for_class(self.label)


@dataclass(frozen=True)
class PPEIncident:
    incident_id: str
    source_id: str
    event_type: str
    label: str
    confidence: float
    detected_at: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class PPEDetector:
    """Run a local Ultralytics PPE checkpoint and normalize its detections."""

    def __init__(self, weights_path: Path):
        from ultralytics import YOLO

        if not weights_path.is_file():
            raise FileNotFoundError(f"PPE model weights not found: {weights_path}")
        self.model = YOLO(str(weights_path))
        names = self.model.names
        self.names = {
            int(class_id): str(name)
            for class_id, name in (
                names.items() if isinstance(names, dict) else enumerate(names)
            )
        }
        self.supported_issues = {
            issue
            for label in self.names.values()
            if (issue := issue_for_class(label)) is not None
        }
        if not self.supported_issues:
            raise ValueError(
                "The selected model has no recognized missing-PPE classes. "
                "Expected labels such as No-Helmet, No-Vest, or No-Boots."
            )

    def detect(self, frame, confidence: float) -> list[PPEDetection]:
        """Return model detections above the requested confidence threshold."""
        results = self.model.predict(frame, conf=confidence, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                coordinates = box.xyxy[0].cpu().numpy().astype(int).tolist()
                detections.append(
                    PPEDetection(
                        label=self.names[class_id],
                        confidence=float(box.conf[0].item()),
                        box=tuple(coordinates),
                    )
                )
        return detections


class PPEAlertValidator:
    """Require persistent detections and suppress repeats during one event."""

    def __init__(
        self,
        consecutive_frames: int = 5,
        clear_frames: int = 12,
        cooldown_seconds: int = 300,
    ):
        if consecutive_frames < 1 or clear_frames < 1 or cooldown_seconds < 0:
            raise ValueError("Frame counts must be positive and cooldown non-negative.")
        self.consecutive_frames = consecutive_frames
        self.clear_frames = clear_frames
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self._consecutive: dict[str, int] = {}
        self._clear_count: dict[str, int] = {}
        self._active: set[str] = set()
        self._last_alert: dict[tuple[str, str], datetime] = {}

    def update(
        self,
        detections: Iterable[PPEDetection],
        source_id: str,
        now: datetime | None = None,
    ) -> list[PPEIncident]:
        current_time = now or datetime.now(timezone.utc)
        confidences: dict[str, float] = {}
        for detection in detections:
            if detection.issue_type:
                confidences[detection.issue_type] = max(
                    confidences.get(detection.issue_type, 0.0),
                    detection.confidence,
                )

        incidents = []
        for issue_type in ISSUE_LABELS:
            if issue_type in confidences:
                self._consecutive[issue_type] = self._consecutive.get(issue_type, 0) + 1
                self._clear_count[issue_type] = 0
                if (
                    self._consecutive[issue_type] >= self.consecutive_frames
                    and issue_type not in self._active
                ):
                    self._active.add(issue_type)
                    cooldown_key = (source_id, issue_type)
                    previous_alert = self._last_alert.get(cooldown_key)
                    if previous_alert is None or current_time - previous_alert >= self.cooldown:
                        incidents.append(
                            PPEIncident(
                                incident_id=str(uuid.uuid4()),
                                source_id=source_id,
                                event_type=issue_type,
                                label=ISSUE_LABELS[issue_type],
                                confidence=confidences[issue_type],
                                detected_at=current_time.isoformat(),
                            )
                        )
                        self._last_alert[cooldown_key] = current_time
            else:
                self._consecutive[issue_type] = 0
                self._clear_count[issue_type] = self._clear_count.get(issue_type, 0) + 1
                if self._clear_count[issue_type] >= self.clear_frames:
                    self._active.discard(issue_type)

        return incidents
