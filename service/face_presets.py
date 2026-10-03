"""Exact preset variants and stream-local, stable synthetic identities."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from service.face_metadata import FaceAttributes

DEFAULT_PRESET_MANIFEST = Path(__file__).resolve().parents[1] / "config" / "face_presets.json"
IDENTITY_COUNT = 5


@dataclass(frozen=True, slots=True)
class FacePreset:
    attributes: FaceAttributes
    identity: int
    image: Path

    @property
    def key(self) -> str:
        a = self.attributes
        return f"{a.gender}/{a.age}/{self.identity}/{a.glasses}/{a.exp}"


class PresetCatalog:
    def __init__(self, manifest: Path):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("presets"), list):
            raise ValueError("invalid face preset manifest")
        self.presets: dict[tuple[FaceAttributes, int], FacePreset] = {}
        root = manifest.resolve().parent
        for entry in payload["presets"]:
            attrs = FaceAttributes(
                **{name: entry[name] for name in ("gender", "age", "glasses", "exp")}
            )
            identity = entry["identity"]
            if type(identity) is not int or not 1 <= identity <= IDENTITY_COUNT:
                raise ValueError("preset identity must be in 1..5")
            relative = Path(entry["image"])
            if relative.is_absolute():
                raise ValueError("preset images must be relative to the manifest")
            image = (root / relative).resolve()
            if not image.is_relative_to(root):
                raise ValueError("preset image escapes the manifest directory")
            key = (attrs, identity)
            if key in self.presets:
                raise ValueError(f"duplicate preset: {key}")
            # Incomplete image sets are allowed; missing files fail closed per face.
            self.presets[key] = FacePreset(attrs, identity, image)

    def match(self, attributes: FaceAttributes, identity: int) -> FacePreset | None:
        return self.presets.get((attributes, identity))


@dataclass(slots=True)
class TrackIdentity:
    gender: str
    age: str
    identity: int
    last_seen: int

    @property
    def key(self) -> str:
        return f"{self.gender}/{self.age}/{self.identity}"


class StreamFaceIdentities:
    """Pin gender/age/slot once; exp/glasses select variants on every fresh frame."""

    def __init__(
        self, *, retention_frames: int = 30, choose: Callable[[int], int] = secrets.randbelow
    ):
        self.retention_frames = retention_frames
        self.choose = choose
        self.tracks: dict[int, TrackIdentity] = {}

    def observe(
        self, track_id: int, attributes: FaceAttributes, frame: int
    ) -> tuple[TrackIdentity, FaceAttributes]:
        state = self.tracks.get(track_id)
        if state is None:
            state = TrackIdentity(
                attributes.gender, attributes.age, self.choose(IDENTITY_COUNT) + 1, frame
            )
            self.tracks[track_id] = state
        state.last_seen = frame
        variant = FaceAttributes(state.gender, state.age, attributes.glasses, attributes.exp)
        return state, variant

    def expire(self, current_ids: set[int], frame: int) -> None:
        for track_id, state in tuple(self.tracks.items()):
            if track_id in current_ids:
                state.last_seen = frame
            elif frame - state.last_seen > self.retention_frames:
                del self.tracks[track_id]

    def clear(self) -> None:
        self.tracks.clear()
