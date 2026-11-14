"""Small shared helpers: error type, HTTP session with retries, safe cache writes."""
from __future__ import annotations

import os
import tempfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DataFetchError(RuntimeError):
    """A data download failed or returned something unusable. The message is meant to be shown to users."""


class DictAccess:
    """Lets a dataclass be read like a dict (obj["key"]) as well as obj.key."""
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


def make_session(retries: int = 3, backoff: float = 1.5) -> requests.Session:
    """A requests session that retries temporary server errors with increasing waits."""
    session = requests.Session()
    retry = Retry(total=retries, connect=retries, read=retries, backoff_factor=backoff,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "FieldShift-RotationExplorer/1.0 (student project)"})
    return session


def atomic_to_csv(obj, path: str, **kwargs) -> None:
    """Write a DataFrame/Series to CSV so a crash mid-write never leaves a half-written cache file."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=folder)
    os.close(fd)
    try:
        obj.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
