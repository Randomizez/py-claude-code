from __future__ import annotations

from uuid import uuid4


def uuid7_string() -> str:
    # Python 3.10 stdlib has no uuid7; a random UUID is sufficient for
    # local ids in this minimal extraction.
    return str(uuid4())
