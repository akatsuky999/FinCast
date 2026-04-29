from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
RETURN_BUILDER_PATH = ROOT / "Data" / "Datasets_rate" / "build_fincast_datasets.py"

spec = importlib.util.spec_from_file_location("fincast_return_builder", RETURN_BUILDER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load shared builder from {RETURN_BUILDER_PATH}")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


OUT_ROOT = ROOT / "Data" / "Datasets_return"


def _build_numeric_price_dataset(price: pd.DataFrame, daily_news: pd.DataFrame) -> pd.DataFrame:
    df = price.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["target_close"] = df["close"]
    df["return_lag1"] = np.log(df["close"].shift(1) / df["close"].shift(2))
    df["open_lag1"] = df["open"].shift(1)
    df["high_lag1"] = df["high"].shift(1)
    df["low_lag1"] = df["low"].shift(1)
    df["close_lag1"] = df["close"].shift(1)
    df["volume_lag1"] = df["volume"].shift(1)
    df["log_volume_lag1"] = np.log1p(df["volume_lag1"])
    df["high_low_range_lag1"] = (df["high"].shift(1) - df["low"].shift(1)) / df["close_lag1"]
    df["open_close_gap_lag1"] = (df["open"].shift(1) - df["close"].shift(2)) / df["close"].shift(2)

    counts = daily_news[["date", "news_count", "analyst_titles_json", "partner_headlines_json"]].copy()
    counts["analyst_news_count"] = counts["analyst_titles_json"].map(lambda x: len(shared.json.loads(x)))
    counts["partner_news_count"] = counts["partner_headlines_json"].map(lambda x: len(shared.json.loads(x)))
    counts = counts[["date", "news_count", "analyst_news_count", "partner_news_count"]]
    df = df.merge(counts, on="date", how="left")
    df[["news_count", "analyst_news_count", "partner_news_count"]] = df[
        ["news_count", "analyst_news_count", "partner_news_count"]
    ].fillna(0)
    df["news_count_lag1"] = df["news_count"].shift(1)
    df["analyst_news_count_lag1"] = df["analyst_news_count"].shift(1)
    df["partner_news_count_lag1"] = df["partner_news_count"].shift(1)

    cols = [
        "date",
        "open_lag1",
        "high_lag1",
        "low_lag1",
        "close_lag1",
        "volume_lag1",
        "log_volume_lag1",
        "return_lag1",
        "high_low_range_lag1",
        "open_close_gap_lag1",
        "news_count_lag1",
        "analyst_news_count_lag1",
        "partner_news_count_lag1",
        "target_close",
    ]
    out = df[cols].dropna().copy()
    numeric_cols = [c for c in out.columns if c != "date"]
    if not np.isfinite(out[numeric_cols].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite values found in numeric price dataset.")
    return out.reset_index(drop=True)


def _write_context(ticker: str, meta: pd.Series | None, full: pd.DataFrame) -> str:
    company = ticker
    sector = "unknown"
    market_cap_group = "unknown"
    if meta is not None:
        company = str(meta.get("company", ticker))
        sector = str(meta.get("sector", "unknown"))
        market_cap_group = str(meta.get("marketCapGroup", "unknown"))
    return (
        f"# FinCast-Price-{ticker} contextual briefing\n\n"
        f"{ticker} represents {company}, a {market_cap_group} capitalization company in the {sector} sector.\n"
        "The forecasting target is the daily closing price. Numeric exogenous variables are one-trading-day lagged OHLCV, lagged log return, intraday range, open-close gap, and lagged news-volume counts.\n"
        "News titles and partner headlines are aligned to the next available trading day to reduce look-ahead leakage. Use daily_news.csv as the textual sidecar when injecting market context into the Briefing agent.\n\n"
        f"Full period: {full['date'].min().date()} to {full['date'].max().date()} ({len(full)} rows).\n"
        "This dataset is intentionally not pre-split. Build train/validation/test samples from rolling windows in the experiment layer.\n"
    )


def _validate_dataset(ticker: str, full: pd.DataFrame, daily_news: pd.DataFrame, price_dates: pd.Series) -> None:
    if list(full.columns)[-1] != "target_close":
        raise RuntimeError(f"{ticker}: target_close is not the last column.")
    if not full["date"].is_monotonic_increasing:
        raise RuntimeError(f"{ticker}: dates are not monotonic.")
    if full["date"].duplicated().any():
        raise RuntimeError(f"{ticker}: duplicate dates found.")
    numeric_cols = [c for c in full.columns if c != "date"]
    if not np.isfinite(full[numeric_cols].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{ticker}: non-finite numeric values found.")
    if len(full) < shared.LOOK_BACK + shared.PREDICTED_WINDOW:
        raise RuntimeError(f"{ticker}: not enough rows for configured look_back/window.")

    valid_dates = set(pd.to_datetime(price_dates).dt.normalize())
    news_dates = set(pd.to_datetime(daily_news["date"]))
    if not news_dates.issubset(valid_dates):
        raise RuntimeError(f"{ticker}: daily_news contains dates outside the trading calendar.")


def _write_manifest() -> None:
    lines = [
        "name: FinCast price full-series benchmark",
        "use_features: True",
        "use_exogenous: True",
        f"look_back: {shared.LOOK_BACK}",
        f"predicted_window: {shared.PREDICTED_WINDOW}",
        f"sliding_window: {shared.SLIDING_WINDOW}",
        "frequency: B",
        "",
        "datasets:",
    ]
    for ticker in shared.TICKERS:
        base = f"FinCast/Data/Datasets_return/{ticker}"
        lines.extend(
            [
                f"  - name: FinCastPrice_{ticker}",
                f"    aliases: [{ticker.lower()}_price, {ticker}_PRICE]",
                f"    full_csv: {base}/full.csv",
                f"    daily_news_csv: {base}/daily_news.csv",
                f"    context_prompt_file: {base}/context.txt",
            ]
        )
    (OUT_ROOT / "manifest_fincast_price.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    old_config = OUT_ROOT / "config_fincast_price.yaml"
    if old_config.exists():
        old_config.unlink()


def _write_figures() -> None:
    fig_dir = OUT_ROOT / "figure"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(shared.TICKERS), 1, figsize=(14, 12), sharex=True)
    for ax, ticker in zip(axes, shared.TICKERS):
        full = pd.read_csv(OUT_ROOT / ticker / "full.csv", parse_dates=["date"])
        ax.plot(full["date"], full["target_close"], linewidth=0.9, color="#2563eb", label="full")
        ax.set_title(f"{ticker} target_close", loc="left", fontsize=11)
        ax.set_ylabel("close")
        ax.legend(loc="upper left", fontsize=8, frameon=True)

        one_fig, one_ax = plt.subplots(figsize=(14, 4))
        one_ax.plot(full["date"], full["target_close"], linewidth=0.9, color="#2563eb", label="full")
        one_ax.set_title(f"FinCast Price {ticker}: target_close full series")
        one_ax.set_xlabel("date")
        one_ax.set_ylabel("daily close")
        one_ax.legend(loc="upper left")
        one_fig.tight_layout()
        one_fig.savefig(fig_dir / f"{ticker}_target_close.png", dpi=180)
        plt.close(one_fig)

    axes[-1].set_xlabel("date")
    fig.suptitle("FinCast price benchmark target_close full series by ticker", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fig_dir / "fincast_target_close_all.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    prices = shared._read_prices()
    news = shared._read_and_align_news(prices)
    meta = pd.read_csv(shared.NASDAQ_LIST_PATH, encoding="latin1")

    summary_rows = []
    for ticker in shared.TICKERS:
        out_dir = OUT_ROOT / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        price = prices[prices["ticker"] == ticker].copy()
        daily_news = shared._daily_news_for_ticker(ticker, price["date"], news)
        data = _build_numeric_price_dataset(price, daily_news)
        daily_news = daily_news[daily_news["date"].isin(set(data["date"]))].reset_index(drop=True)

        meta_row = meta[meta["ticker"] == ticker]
        context = _write_context(ticker, meta_row.iloc[0] if not meta_row.empty else None, data)
        _validate_dataset(ticker, data, daily_news, price["date"])

        data.to_csv(out_dir / "full.csv", index=False)
        daily_news.to_csv(out_dir / "daily_news.csv", index=False)
        (out_dir / "context.txt").write_text(context, encoding="utf-8")
        for old_name in ("train.csv", "test.csv"):
            old_path = out_dir / old_name
            if old_path.exists():
                old_path.unlink()

        summary_rows.append(
            {
                "ticker": ticker,
                "full_rows": len(data),
                "news_days": int((daily_news["news_count"] > 0).sum()),
                "news_items": int(daily_news["news_count"].sum()),
                "target_min": float(data["target_close"].min()),
                "target_max": float(data["target_close"].max()),
                "start": data["date"].min().date().isoformat(),
                "end": data["date"].max().date().isoformat(),
            }
        )

    _write_manifest()
    _write_figures()
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_ROOT / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote manifest: {OUT_ROOT / 'manifest_fincast_price.yaml'}")


if __name__ == "__main__":
    main()
