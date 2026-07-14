from __future__ import annotations

import os


def load_hf_token(required: bool = False) -> str | None:
    """Read a token without printing it or writing it to run metadata."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = token
        return token
    if required:
        raise RuntimeError(
            "HF_TOKEN is unavailable. Add it to Kaggle Secrets and enable notebook access."
        )
    return None

