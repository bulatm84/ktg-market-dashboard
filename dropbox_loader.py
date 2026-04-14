"""
Dropbox file loader for Streamlit Cloud deployment.
Downloads CSV/parquet files from Dropbox using a refresh token.
Falls back to local files when running locally.
"""

import io
import streamlit as st
import pandas as pd
import dropbox


DROPBOX_FOLDER = "/Python/SMB Python/Rader Joint"


def _get_client():
    """Create a Dropbox client using refresh token from Streamlit secrets."""
    return dropbox.Dropbox(
        oauth2_refresh_token=st.secrets["dropbox"]["refresh_token"],
        app_key=st.secrets["dropbox"]["app_key"],
        app_secret=st.secrets["dropbox"]["app_secret"],
    )


@st.cache_data(ttl=3600)
def _download_bytes(filename: str) -> bytes:
    """Download a file from Dropbox and return raw bytes. Cached 1 hour."""
    dbx = _get_client()
    path = f"{DROPBOX_FOLDER}/{filename}"
    _, response = dbx.files_download(path)
    return response.content


def read_csv(filename: str, **kwargs) -> pd.DataFrame:
    """Read a CSV from Dropbox."""
    raw = _download_bytes(filename)
    return pd.read_csv(io.BytesIO(raw), **kwargs)


def read_parquet(filename: str, **kwargs) -> pd.DataFrame:
    """Read a parquet from Dropbox."""
    raw = _download_bytes(filename)
    return pd.read_parquet(io.BytesIO(raw), **kwargs)
