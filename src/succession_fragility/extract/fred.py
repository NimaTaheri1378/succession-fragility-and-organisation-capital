from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class FredClient:
    api_key: str | None = None

    def key(self) -> str:
        key = self.api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise RuntimeError("Set FRED_API_KEY before pulling regime controls.")
        return key

    def series_observations(
        self,
        series_id: str,
        start: str = "1963-01-01",
        end: str | None = None,
        allow_keyless_csv: bool = True,
    ) -> pd.DataFrame:
        """Pull a FRED series without ever writing an API key.

        If ``FRED_API_KEY`` is available we use the official JSON API. Otherwise
        we fall back to FRED's public graph CSV endpoint, which is enough for the
        regime classifications used in this project and avoids storing secrets.
        """

        key = self.api_key or os.environ.get("FRED_API_KEY")
        if not key and allow_keyless_csv:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df = pd.read_csv(url)
            date_col = df.columns[0]
            value_col = series_id if series_id in df.columns else df.columns[-1]
            df = df.rename(columns={date_col: "date", value_col: "value"})
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
            out = df[["date", "value"]]
            out = out[out["date"] >= pd.Timestamp(start)]
            if end is not None:
                out = out[out["date"] <= pd.Timestamp(end)]
            return out.reset_index(drop=True)

        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": key or self.key(),
            "file_type": "json",
            "observation_start": start,
        }
        if end is not None:
            params["observation_end"] = end
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        obs = resp.json()["observations"]
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]]


REGIME_SERIES = {
    "vix": "VIXCLS",
    "nber_recession": "USREC",
    "fed_funds": "FEDFUNDS",
    "term_spread": "T10Y3M",
    "credit_spread": "BAA10YM",
}
