from dataclasses import dataclass


@dataclass(frozen=True)
class ShakeWarning:
    code: str
    message: str
    section: tuple[str, ...] | None = None
