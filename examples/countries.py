# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "lxml", "cachetto"]
# ///

"""Run with:

uv run examples/countries.py
"""

from time import time

import pandas as pd

from cachetto import cached


@cached
def get_countries() -> pd.DataFrame:
    dfs = pd.read_html(
        r"https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
    )
    df = dfs[2]  # Select list of countries
    df.columns = ["_".join(col).strip().replace(" ", "_") for col in df.columns.values]
    # Cleanup columns
    df = df.rename(
        columns={
            "Country/Territory_Country/Territory": "Country",
            "IMF[1][12]_Forecast": "IMF_Forecast",
            "IMF[1][12]_Year": "IMF_Year",
            "World_Bank[13]_Estimate": "WB_Estimate",
            "World_Bank[13]_Year": "WB_Year",
            "United_Nations[14]_Estimate": "UN_Estimate",
            "United_Nations[14]_Year": "UN_Year",
        }
    )
    # Cast numeric columns and clean them
    for col in ["IMF_Forecast", "WB_Estimate", "UN_Estimate"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce"
        )
        df[col] = df[col].replace(r"[^\d.]", "", regex=True).astype(float)
    # Calculate GDP growth
    df["IMF_Growth(%)"] = (
        (df["IMF_Forecast"] - df["WB_Estimate"]) / df["WB_Estimate"] * 100
    )
    df = df.sort_values("IMF_Forecast", ascending=False)
    return df


print("Start downloading...")
t0 = time()
print(get_countries().head(5))
print(f"Seconds elapsed: {time() - t0:.4f}")
print("\n\n")
print("Call again...")
t1 = time()
print(get_countries().head(5))
print(f"Seconds elapsed: {time() - t1:.4f}")
