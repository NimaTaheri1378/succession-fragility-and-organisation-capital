from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class SecClient:
    user_agent: str | None = None
    sleep_seconds: float = 0.12

    def headers(self) -> dict[str, str]:
        ua = self.user_agent or os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise RuntimeError("Set SEC_USER_AGENT before using SEC EDGAR endpoints.")
        return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

    def submissions(self, cik: str) -> dict[str, Any]:
        cik10 = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        resp = requests.get(url, headers=self.headers(), timeout=30)
        time.sleep(self.sleep_seconds)
        resp.raise_for_status()
        return resp.json()

    def submissions_file(self, file_name: str) -> dict[str, Any]:
        url = f"https://data.sec.gov/submissions/{file_name}"
        resp = requests.get(url, headers=self.headers(), timeout=30)
        time.sleep(self.sleep_seconds)
        resp.raise_for_status()
        return resp.json()


def recent_filings_for_forms(payload: dict[str, Any], forms: set[str]) -> list[dict[str, Any]]:
    recent = payload.get("filings", {}).get("recent", {})
    out: list[dict[str, Any]] = []
    for idx, form in enumerate(recent.get("form", [])):
        if form not in forms:
            continue
        out.append(
            {
                "form": form,
                "filing_date": recent.get("filingDate", [None])[idx],
                "acceptance_datetime": recent.get("acceptanceDateTime", [None])[idx],
                "accession_number": recent.get("accessionNumber", [None])[idx],
                "primary_document": recent.get("primaryDocument", [None])[idx],
            }
        )
    return out


def filing_rows_for_forms(payload: dict[str, Any], forms: set[str]) -> list[dict[str, Any]]:
    """Parse SEC submissions JSON in both recent and historical-file formats."""

    if "filings" in payload:
        recent = payload.get("filings", {}).get("recent", {})
    else:
        recent = payload
    out: list[dict[str, Any]] = []
    form_values = recent.get("form", [])
    for idx, form in enumerate(form_values):
        if form not in forms:
            continue
        out.append(
            {
                "form": form,
                "filing_date": recent.get("filingDate", [None])[idx],
                "acceptance_datetime": recent.get("acceptanceDateTime", [None])[idx],
                "accession_number": recent.get("accessionNumber", [None])[idx],
                "primary_document": recent.get("primaryDocument", [None])[idx],
            }
        )
    return out


def historical_submission_file_names(payload: dict[str, Any]) -> list[str]:
    return [item.get("name") for item in payload.get("filings", {}).get("files", []) if item.get("name")]
