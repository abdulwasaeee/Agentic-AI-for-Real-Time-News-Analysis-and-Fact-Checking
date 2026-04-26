"""Unique ID generation helpers."""

import uuid


def make_id(prefix: str = "") -> str:
    """Generate a short unique ID with an optional prefix."""
    return f"{prefix}{uuid.uuid4().hex[:12]}"
