from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TICKERS = ("NVDA", "NFLX")
START_DATE = pd.Timestamp("2010-01-01")
END_DATE = pd.Timestamp("2020-02-06")
TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")

LOOK_BACK = 60
PREDICTED_WINDOW = 5
SLIDING_WINDOW = 5

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "Raw_data"
OUT_ROOT = ROOT / "Data" / "Datasets_rate"

PRICE_PATH = (
    RAW_ROOT
    / "6000+ Nasdaq Stocks Historical Daily Prices"
    / "nasdaq_historical_prices_daily"
    / "nasdaq_historical_prices_daily.csv"
)
NASDAQ_LIST_PATH = RAW_ROOT / "6000+ Nasdaq Stocks Historical Daily Prices" / "nasdaq_list.csv"
ANALYST_NEWS_PATH = RAW_ROOT / "Daily Financial News for 6000+ Stocks" / "analyst_ratings_processed.csv"
PARTNER_NEWS_PATH = RAW_ROOT / "Daily Financial News for 6000+ Stocks" / "raw_partner_headlines.csv"


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return " ".join(text.split())


def _read_prices() -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    usecols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    for chunk in pd.read_csv(PRICE_PATH, usecols=usecols, parse_dates=["date"], chunksize=1_000_000):
        mask = (
            chunk["ticker"].isin(TICKERS)
            & (chunk["date"] >= START_DATE)
            & (chunk["date"] <= END_DATE)
        )
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        raise RuntimeError("No price rows found for selected tickers and date range.")
    prices = pd.concat(chunks, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return prices.reset_index(drop=True)


def _next_trading_day(day: pd.Timestamp, trading_days: list[pd.Timestamp]) -> pd.Timestamp | None:
    idx = bisect_left(trading_days, day.normalize())
    if idx >= len(trading_days):
        return None
    candidate = trading_days[idx]
    if candidate < day.normalize():
        return None
    return candidate


def _aggregate_titles(rows: Iterable[str], max_items: int = 20) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in rows:
        text = _clean_text(raw)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _read_and_align_news(prices: pd.DataFrame) -> pd.DataFrame:
    trading_days_by_ticker = {
        ticker: list(group["date"].sort_values())
        for ticker, group in prices.groupby("ticker", sort=False)
    }

    aligned_frames: list[pd.DataFrame] = []

    analyst = pd.read_csv(ANALYST_NEWS_PATH, usecols=["title", "date", "stock"])
    analyst = analyst[analyst["stock"].isin(TICKERS)].copy()
    analyst["published_at"] = pd.to_datetime(analyst["date"], utc=True, errors="coerce").dt.tz_convert(None)
    analyst = analyst.dropna(subset=["published_at"])
    analyst = analyst[(analyst["published_at"] >= START_DATE) & (analyst["published_at"] <= END_DATE)]
    analyst["source"] = "analyst"
    analyst["text"] = analyst["title"].map(_clean_text)
    analyst = analyst.rename(columns={"stock": "ticker"})
    aligned_frames.append(analyst[["ticker", "published_at", "source", "text"]])

    partner = pd.read_csv(PARTNER_NEWS_PATH, usecols=["headline", "date", "stock", "publisher"])
    partner = partner[partner["stock"].isin(TICKERS)].copy()
    partner["published_at"] = pd.to_datetime(partner["date"], utc=True, errors="coerce").dt.tz_convert(None)
    partner = partner.dropna(subset=["published_at"])
    partner = partner[(partner["published_at"] >= START_DATE) & (partner["published_at"] <= END_DATE)]
    partner["source"] = "partner"
    partner["publisher"] = partner["publisher"].map(_clean_text)
    partner["headline"] = partner["headline"].map(_clean_text)
    partner["text"] = np.where(
        partner["publisher"].eq(""),
        partner["headline"],
        partner["publisher"] + ": " + partner["headline"],
    )
    partner = partner.rename(columns={"stock": "ticker"})
    aligned_frames.append(partner[["ticker", "published_at", "source", "text"]])

    news = pd.concat(aligned_frames, ignore_index=True)
    news = news[news["text"].astype(bool)].copy()
    effective_dates: list[pd.Timestamp | None] = []
    for row in news.itertuples(index=False):
        days = trading_days_by_ticker.get(row.ticker, [])
        effective_dates.append(_next_trading_day(pd.Timestamp(row.published_at), days))
    news["date"] = effective_dates
    news = news.dropna(subset=["date"]).copy()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    return news


def _daily_news_for_ticker(ticker: str, price_dates: pd.Series, news: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({"date": pd.to_datetime(price_dates).dt.normalize()})
    calendar["ticker"] = ticker

    ticker_news = news[news["ticker"] == ticker].copy()
    if ticker_news.empty:
        calendar["news_count"] = 0
        calendar["analyst_titles_json"] = "[]"
        calendar["partner_headlines_json"] = "[]"
        calendar["combined_news_text"] = ""
        return calendar

    grouped_rows = []
    for day, group in ticker_news.groupby("date", sort=True):
        analyst_titles = _aggregate_titles(group.loc[group["source"] == "analyst", "text"])
        partner_headlines = _aggregate_titles(group.loc[group["source"] == "partner", "text"])
        combined = analyst_titles + partner_headlines
        grouped_rows.append(
            {
                "date": day,
                "ticker": ticker,
                "news_count": int(len(group)),
                "analyst_titles_json": json.dumps(analyst_titles, ensure_ascii=False),
                "partner_headlines_json": json.dumps(partner_headlines, ensure_ascii=False),
                "combined_news_text": " | ".join(combined),
            }
        )

    daily = pd.DataFrame(grouped_rows)
    out = calendar.merge(daily, on=["date", "ticker"], how="left")
    out["news_count"] = out["news_count"].fillna(0).astype(int)
    out["analyst_titles_json"] = out["analyst_titles_json"].fillna("[]")
    out["partner_headlines_json"] = out["partner_headlines_json"].fillna("[]")
    out["combined_news_text"] = out["combined_news_text"].fillna("")
    return out


def _build_numeric_dataset(price: pd.DataFrame, daily_news: pd.DataFrame) -> pd.DataFrame:
    df = price.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["target_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_lag1"] = df["target_return"].shift(1)
    df["open_lag1"] = df["open"].shift(1)
    df["high_lag1"] = df["high"].shift(1)
    df["low_lag1"] = df["low"].shift(1)
    df["close_lag1"] = df["close"].shift(1)
    df["volume_lag1"] = df["volume"].shift(1)
    df["log_volume_lag1"] = np.log1p(df["volume_lag1"])
    df["high_low_range_lag1"] = (df["high"].shift(1) - df["low"].shift(1)) / df["close_lag1"]
    df["open_close_gap_lag1"] = (df["open"].shift(1) - df["close"].shift(2)) / df["close"].shift(2)

    counts = daily_news[["date", "news_count", "analyst_titles_json", "partner_headlines_json"]].copy()
    counts["analyst_news_count"] = counts["analyst_titles_json"].map(lambda x: len(json.loads(x)))
    counts["partner_news_count"] = counts["partner_headlines_json"].map(lambda x: len(json.loads(x)))
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
        "target_return",
    ]
    out = df[cols].dropna().copy()
    numeric_cols = [c for c in out.columns if c != "date"]
    if not np.isfinite(out[numeric_cols].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite values found in numeric dataset.")
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
        f"# FinCast-{ticker} contextual briefing\n\n"
        f"{ticker} represents {company}, a {market_cap_group} capitalization company in the {sector} sector.\n"
        "The forecasting target is daily log return of the stock close price. Numeric exogenous variables are one-trading-day lagged OHLCV, lagged return, intraday range, open-close gap, and lagged news-volume counts.\n"
        "News titles and partner headlines are aligned to the next available trading day to reduce look-ahead leakage. Use daily_news.csv as the textual sidecar when injecting market context into the Briefing agent.\n\n"
        f"Full period: {full['date'].min().date()} to {full['date'].max().date()} ({len(full)} rows).\n"
        "This dataset is intentionally not pre-split. Build train/validation/test samples from rolling windows in the experiment layer.\n"
    )


def _validate_dataset(ticker: str, full: pd.DataFrame, daily_news: pd.DataFrame, price_dates: pd.Series) -> None:
    if list(full.columns)[-1] != "target_return":
        raise RuntimeError(f"{ticker}: target_return is not the last column.")
    if not full["date"].is_monotonic_increasing:
        raise RuntimeError(f"{ticker}: dates are not monotonic.")
    if full["date"].duplicated().any():
        raise RuntimeError(f"{ticker}: duplicate dates found.")
    numeric_cols = [c for c in full.columns if c != "date"]
    if not np.isfinite(full[numeric_cols].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{ticker}: non-finite numeric values found.")
    if len(full) < LOOK_BACK + PREDICTED_WINDOW:
        raise RuntimeError(f"{ticker}: not enough rows for configured look_back/window.")

    valid_dates = set(pd.to_datetime(price_dates).dt.normalize())
    news_dates = set(pd.to_datetime(daily_news["date"]))
    if not news_dates.issubset(valid_dates):
        raise RuntimeError(f"{ticker}: daily_news contains dates outside the trading calendar.")


def _write_manifest() -> None:
    lines = [
        "name: FinCast return full-series benchmark",
        "use_features: True",
        "use_exogenous: True",
        f"look_back: {LOOK_BACK}",
        f"predicted_window: {PREDICTED_WINDOW}",
        f"sliding_window: {SLIDING_WINDOW}",
        "frequency: B",
        "",
        "datasets:",
    ]
    for ticker in TICKERS:
        base = f"FinCast/Data/Datasets_rate/{ticker}"
        lines.extend(
            [
                f"  - name: FinCast_{ticker}",
                f"    aliases: [{ticker.lower()}, {ticker}]",
                f"    full_csv: {base}/full.csv",
                f"    daily_news_csv: {base}/daily_news.csv",
                f"    context_prompt_file: {base}/context.txt",
            ]
        )
    (OUT_ROOT / "manifest_fincast_return.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    old_config = OUT_ROOT / "config_fincast_small.yaml"
    if old_config.exists():
        old_config.unlink()


def _write_figures() -> None:
    fig_dir = OUT_ROOT / "figure"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(TICKERS), 1, figsize=(14, 12), sharex=True)
    for ax, ticker in zip(axes, TICKERS):
        full = pd.read_csv(OUT_ROOT / ticker / "full.csv", parse_dates=["date"])
        ax.plot(full["date"], full["target_return"], linewidth=0.7, color="#2563eb", label="full")
        ax.axhline(0, color="#111827", linewidth=0.7, alpha=0.6)
        ax.set_title(f"{ticker} target_return", loc="left", fontsize=11)
        ax.set_ylabel("log return")
        ax.legend(loc="upper right", fontsize=8, frameon=True)

        one_fig, one_ax = plt.subplots(figsize=(14, 4))
        one_ax.plot(full["date"], full["target_return"], linewidth=0.7, color="#2563eb", label="full")
        one_ax.axhline(0, color="#111827", linewidth=0.7, alpha=0.6)
        one_ax.set_title(f"FinCast {ticker}: target_return full series")
        one_ax.set_xlabel("date")
        one_ax.set_ylabel("daily log return")
        one_ax.legend(loc="upper right")
        one_fig.tight_layout()
        one_fig.savefig(fig_dir / f"{ticker}_target_return.png", dpi=180)
        plt.close(one_fig)

    axes[-1].set_xlabel("date")
    fig.suptitle("FinCast return benchmark target_return full series by ticker", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fig_dir / "fincast_target_return_all.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    prices = _read_prices()
    news = _read_and_align_news(prices)
    meta = pd.read_csv(NASDAQ_LIST_PATH, encoding="latin1")

    summary_rows = []
    for ticker in TICKERS:
        out_dir = OUT_ROOT / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        price = prices[prices["ticker"] == ticker].copy()
        daily_news = _daily_news_for_ticker(ticker, price["date"], news)
        data = _build_numeric_dataset(price, daily_news)
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
                "start": data["date"].min().date().isoformat(),
                "end": data["date"].max().date().isoformat(),
            }
        )

    _write_manifest()
    _write_figures()
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_ROOT / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote manifest: {OUT_ROOT / 'manifest_fincast_return.yaml'}")


if __name__ == "__main__":
    main()
