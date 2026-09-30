"""
FastAPI lookup service for an authorized/synthetic Hugging Face bucket.

Endpoints:
  GET /number=<phone>
  GET /aadhar=<aadhaar>

Example:
  /number=9876543210
  /aadhar=123456789012

The service uses the pre-sorted idx_phone.*.parquet and
idx_aadhaar.*.parquet parts directly through Hugging Face + DuckDB.
It does NOT download the full bucket at startup.

For a public bucket no HF_TOKEN is required.
For a private bucket, set HF_TOKEN in the environment.

IMPORTANT:
Only expose this service when you are authorized to query the data.
For real Aadhaar/phone data, keep the API private/authenticated.
"""

import os
import re
import time
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException

try:
    from huggingface_hub import HfFileSystem
except ImportError as exc:
    raise RuntimeError("Install huggingface_hub") from exc


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BUCKET = os.getenv(
    "HF_BUCKET",
    "hf://buckets/CutehackX/icrm-hitek-full-db-mixed-bucket",
)

# Leave empty for a public bucket.
HF_TOKEN = os.getenv("HF_TOKEN") or None

# Keep this modest on Render Free. Increase only after testing.
DUCKDB_THREADS = int(os.getenv("DUCKDB_THREADS", "4"))

# Maximum rows returned by one lookup.
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "25"))
DEVELOPER = "chatpataprani"

# Simple input limits.
PHONE_RE = re.compile(r"^\d{10,15}$")
AADHAAR_RE = re.compile(r"^\d{12}$")

app = FastAPI(
    title="ICMR/HITEK Fast Lookup API",
    version="2.0",
    description="Fast lookup API — Developer: chatpataprani",
)


# ---------------------------------------------------------------------
# Hugging Face filesystem
# ---------------------------------------------------------------------

fs = HfFileSystem(token=HF_TOKEN)

# Discover the split index files ONCE when the process starts.
# This avoids listing the bucket on every API request.
PHONE_FILES = sorted(
    fs.glob(f"{BUCKET}/idx_phone.*.parquet")
)

AADHAAR_FILES = sorted(
    fs.glob(f"{BUCKET}/idx_aadhaar.*.parquet")
)

# Some versions/datasets use "idx_aadhar" instead of "idx_aadhaar".
# Fall back to that spelling if needed.
if not AADHAAR_FILES:
    AADHAAR_FILES = sorted(
        fs.glob(f"{BUCKET}/idx_aadhar.*.parquet")
    )


def _sql_files(files: list[str]) -> str:
    if not files:
        return ""
    return ", ".join("'" + f.replace("'", "''") + "'" for f in files)


PHONE_LIST = _sql_files(PHONE_FILES)
AADHAAR_LIST = _sql_files(AADHAAR_FILES)

if not PHONE_FILES:
    raise RuntimeError("No idx_phone.*.parquet files were found in the bucket.")

if not AADHAAR_FILES:
    raise RuntimeError(
        "No idx_aadhaar.*.parquet or idx_aadhar.*.parquet files were found."
    )


# ---------------------------------------------------------------------
# DuckDB connection
# ---------------------------------------------------------------------

con = duckdb.connect(database=":memory:")

# Register Hugging Face's fsspec filesystem with DuckDB.
duckdb.register_filesystem(fs)

con.execute(f"SET threads = {DUCKDB_THREADS}")

# Views are cheap: DuckDB reads only the Parquet row groups it needs.
con.execute(
    f"""
    CREATE VIEW phone_index AS
    SELECT *
    FROM read_parquet([{PHONE_LIST}])
    """
)

con.execute(
    f"""
    CREATE VIEW aadhaar_index AS
    SELECT *
    FROM read_parquet([{AADHAAR_LIST}])
    """
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _rows(result) -> list[dict[str, Any]]:
    columns = [d[0] for d in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def _clean_value(value: str) -> str:
    # Parameter binding is used for values; this is only for logging/errors.
    return value.strip()


def _run_exact(view: str, column: str, value: str) -> list[dict[str, Any]]:
    started = time.perf_counter()

    result = con.execute(
        f"""
        SELECT *
        FROM {view}
        WHERE CAST({column} AS VARCHAR) = ?
        LIMIT ?
        """,
        [value, MAX_RESULTS],
    )

    rows = _rows(result)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    return {
        "results": rows,
        "count": len(rows),
        "lookup_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "developer": DEVELOPER,
        "endpoints": {
            "number": "/number=<10-15 digit number>",
            "aadhar": "/aadhar=<12 digit Aadhaar>",
        },
        "phone_index_parts": len(PHONE_FILES),
        "aadhar_index_parts": len(AADHAAR_FILES),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "developer": DEVELOPER,
        "phone_index_parts": len(PHONE_FILES),
        "aadhar_index_parts": len(AADHAAR_FILES),
    }


@app.get("/number={number}")
def number_lookup(number: str):
    number = _clean_value(number)

    if not PHONE_RE.fullmatch(number):
        raise HTTPException(
            status_code=400,
            detail="Number must contain 10 to 15 digits.",
        )

    data = _run_exact(
        "phone_index",
        "phoneNumber",
        number,
    )

    return {
        "status": "success" if data["count"] else "not_found",
        "developer": DEVELOPER,
        "type": "number",
        "number": number,
        **data,
    }


@app.get("/aadhar={aadhar}")
def aadhar_lookup(aadhar: str):
    aadhar = _clean_value(aadhar)

    if not AADHAAR_RE.fullmatch(aadhar):
        raise HTTPException(
            status_code=400,
            detail="Aadhaar value must contain exactly 12 digits.",
        )

    data = _run_exact(
        "aadhaar_index",
        "aadharNumber",
        aadhar,
    )

    return {
        "status": "success" if data["count"] else "not_found",
        "developer": DEVELOPER,
        "type": "aadhar",
        "aadhar": aadhar,
        **data,
    }
