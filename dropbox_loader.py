"""
Dropbox file loader for Streamlit Cloud deployment.
Downloads CSV/parquet files from Dropbox using a refresh token.
Falls back to local files when running locally.
"""

import io
import os
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


def _download_bytes(filename: str) -> bytes:
    """Download a file from Dropbox and return raw bytes."""
    dbx = _get_client()
    path = f"{DROPBOX_FOLDER}/{filename}"
    _, response = dbx.files_download(path)
    return response.content


@st.cache_data(ttl=3600)
def load_csv(filename: str, parse_dates=None) -> pd.DataFrame:
    """Load a CSV from Dropbox (or local fallback) with 1-hour cache."""
    local_path = os.path.join(os.path.dirname(__file__), filename)

    if os.path.exists(local_path) and "dropbox" not in st.secrets:
        # Local development — read from disk
        return pd.read_csv(local_path, parse_dates=parse_dates)

    # Streamlit Cloud — fetch from Dropbox
    raw = _download_bytes(filename)
    return pd.read_csv(io.BytesIO(raw), parse_dates=parse_dates)


@st.cache_data(ttl=3600)
def load_parquet(filename: str) -> pd.DataFrame:
    """Load a parquet from Dropbox (or local fallback) with 1-hour cache."""
    local_path = os.path.join(os.path.dirname(__file__), filename)

    if os.path.exists(local_path) and "dropbox" not in st.secrets:
        return pd.read_parquet(local_path)

    raw = _download_bytes(filename)
    return pd.read_parquet(io.BytesIO(raw))
