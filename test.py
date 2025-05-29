"""

Ziel dieses Skript ist es Aktien zu kristallisieren, die >15-20 % Upside haben

"""

import os
import asyncio
import httpx
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from loguru import logger
import argparse

# Input Kennzahlen Risikofreier Zinsatz etc zu späteren Bewertung
RISK_FREE = 0.025
MARKET_RISK_PREMIUM = 0.06
GLOBAL_TAX_RATE = 0.25
TERMINAL_GROWTH = 0.02

WEIGHTS = {
    "value": 0.4,
    "quality": 0.3,
    "risk": 0.2,
    "momentum": 0.1,
}

MAX_CONCURRENCY = 10  # Anzahl gleichzeitiger Abfragen/ Analysen
TIMEOUT = 60  # Sekunden -> Laufzeit einer Analyse, bei Überschreitung = TimeOut

FMP_API = "https://financialmodelingprep.com/api/v3"  # Je nach Ticker holt sich der Code das Statement automatisiert von der Quelle rein: /income-statement/AAPL?limit=5
API_KEY = os.getenv(
    "Gd7U96FaiCNrPmtb2FpKxIXHsXNXeg8r")  # API Key ist personalisiert zu eurem Daten Account bei FMP, muss also ausgetauscht werden, Anmeldung hier -> https://financialmodelingprep.com/developer/docs


# Daten Input aus Quelle
async def fetch_json(client: httpx.AsyncClient, url: str, params=None):
    try:
        r = await client.get(url, params=params or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Fetch error for {url}: {e}")
        return []


async def get_financials(ticker: str, client: httpx.AsyncClient):
    params = {"apikey": API_KEY}
    inc = await fetch_json(client, f"{FMP_API}/income-statement/{ticker}", {**params, "limit": 5})
    bal = await fetch_json(client, f"{FMP_API}/balance-sheet-statement/{ticker}", {**params, "limit": 5})
    cf = await fetch_json(client, f"{FMP_API}/cash-flow-statement/{ticker}", {**params, "limit": 5})
    prof = await fetch_json(client, f"{FMP_API}/profile/{ticker}", params)
    q = await fetch_json(client, f"{FMP_API}/quote/{ticker}", params)
    return {
        "income": pd.DataFrame(inc),
        "balance": pd.DataFrame(bal),
        "cash": pd.DataFrame(cf),
        "profile": prof[0] if prof else {},
        "quote": q[0] if q else {},
    }


# DCF Modell Berechnung und Wert Berechnung
class ForecastAssumptions:
    def __init__(self, fs):
        inc = fs["income"].head(3)
        self.g1 = float(inc["revenue"].pct_change().mean(skipna=True)) or 0.05
        self.g23 = self.g1 * 0.8
        self.margin = float((inc["operatingIncome"] / inc["revenue"]).median()) or 0.1
        self.tax = GLOBAL_TAX_RATE
        self.dep_pct = float((fs["cash"]["depreciationAndAmortization"] / inc["revenue"]).median()) or 0.03
        self.capex_pct = float((fs["cash"]["capitalExpenditure"].abs() / inc["revenue"]).median()) or 0.02
        nwc = fs["balance"]["totalCurrentAssets"] - fs["balance"]["totalCurrentLiabilities"]
        self.nwc_pct = float((nwc / inc["revenue"]).median()) * 0.02 or 0.01
        self.tg = TERMINAL_GROWTH


def calc_roic(income, balance):
    try:
        nopat = income["operatingIncome"].iloc[0] * (1 - GLOBAL_TAX_RATE)
        invested = balance["totalAssets"].iloc[0] - balance["totalCurrentLiabilities"].iloc[0]
        return nopat / invested
    except:
        return np.nan


def calc_debt_ebitda(balance, income):
    try:
        debt = balance["totalDebt"].iloc[0]
        ebitda = income["ebitda"].iloc[0]
        return debt / ebitda if ebitda else np.nan
    except:
        return np.nan


def calc_momentum(hist_df):
    try:
        returns = hist_df["adjClose"].pct_change().add(1).cumprod().iloc[-1] - 1
        return float(returns)
    except:
        return np.nan


# Bewertung & Zukunfts Erwartung
def calc_wacc(beta, mcap, net_debt):
    ke = RISK_FREE + beta * MARKET_RISK_PREMIUM
    kd = 0.05 * (1 - GLOBAL_TAX_RATE)
    e, d = mcap, max(net_debt, 0)
    return (e / (e + d)) * ke + (d / (e + d)) * kd


def quick_dcf(fs, assum):
    prof = fs["profile"]
    quote = fs["quote"]
    price = quote.get("price", np.nan)
    mcap = prof.get("mktCap", 0)
    shares = mcap / price if price else np.nan
    beta = prof.get("beta", 1.0)
    net_debt = prof.get("debt", 0)
    wacc = calc_wacc(beta, mcap, net_debt)

    rev0 = fs["income"]["revenue"].iloc[0]
    fcfs = []
    rev = rev0 * (1 + assum.g1)
    for yr in range(1, 6):
        if yr > 1:
            rev *= (1 + assum.g23)
        ebit = rev * assum.margin
        nopat = ebit * (1 - assum.tax)
        dep = rev * assum.dep_pct
        capex = rev * assum.capex_pct
        d_nwc = rev * assum.nwc_pct
        fcf = nopat + dep - capex - d_nwc
        fcfs.append(fcf)
    pv_fcfs = [fcf / (1 + wacc) ** t for t, fcf in enumerate(fcfs, start=1)]
    tv = fcfs[-1] * (1 + assum.tg) / (wacc - assum.tg)
    pv_tv = tv / (1 + wacc) ** 5
    ev = sum(pv_fcfs) + pv_tv
    eq_val = ev - net_debt
    fv = eq_val / shares if shares else np.nan
    upside = fv / price - 1 if price else np.nan
    return fv, upside


# hier werden die ausgewerteten Titel gerankt, habe jetzt erstmal nach Signalen gemacht, je nach dem wie die erwartete Rendite im Schritt davor ausfällt
def score_df(df):
    df["value_rank"] = df["upside"].rank(pct=True)
    df["quality_rank"] = df["roic"].rank(pct=True)
    df["risk_rank"] = (1 / df["debt_ebitda"]).rank(pct=True)
    df["mom_rank"] = df["momentum"].rank(pct=True)
    df["score"] = (
            WEIGHTS["value"] * df.value_rank +
            WEIGHTS["quality"] * df.quality_rank +
            WEIGHTS["risk"] * df.risk_rank +
            WEIGHTS["momentum"] * df.mom_rank
    )
    return df


async def process_ticker(tk: str, client: httpx.AsyncClient):  # Verlinkung zum FMP Server
    fs = await get_financials(tk, client)
    assum = ForecastAssumptions(fs)
    fv, upside = quick_dcf(fs, assum)
    roic = calc_roic(fs["income"], fs["balance"])
    debt_ebitda = calc_debt_ebitda(fs["balance"], fs["income"])
    params = {"apikey": API_KEY, "timeseries": 126}
    hist = await fetch_json(client, f"{FMP_API}/historical-price-full/{tk}", params)
    hist_df = pd.DataFrame(hist.get("historical", []))
    momentum = calc_momentum(hist_df)
    return {
        "ticker": tk,
        "fair_value": fv,
        "upside": upside,
        "roic": roic,
        "debt_ebitda": debt_ebitda,
        "momentum": momentum
    }


async def batch_run(tickers):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = [process_ticker(tk, client) for tk in tickers]
        return await tqdm_asyncio.gather(*tasks, limit=MAX_CONCURRENCY)


# Backtasting, aber nicht sicher ob so sinnvoll, denke wäre besser die erwartete Rendite der letzten Jahre x zu testen
def run_backtest(signals, index="^GSPC"):
    return pd.DataFrame()


# Hauptbereich -> Output generation
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="apfel.xlsx",
                        help="...")  # Excel habe ich noch nicht erstellt, Name muss dann hier rein
    parser.add_argument("--sheet", default="Input", help="...")  # Excelname Sheet1 ändern, aktuell auf Input
    parser.add_argument("--tickers-col", default="Ticker", help="...")  # Spalten Überschrift
    parser.add_argument("--out", default="results.xlsx",
                        help="...")  # Excel wird automatisch erstellt mit den folgenden sheets und im selben Ordner abgelegt
    parser.add_argument("--backtest", action="store_true", help="...")
    args = parser.parse_args()

    tickers = pd.read_excel(args.input, sheet_name=args.sheet)[args.tickers_col].dropna().astype(str).tolist()
    df = pd.DataFrame(asyncio.run(batch_run(tickers)))
    df = score_df(df)
    df["buy_signal"] = df["upside"] >= 0.20

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df[["ticker", "fair_value", "upside", "score", "buy_signal"]].to_excel(writer, sheet_name="Results",
                                                                               index=False)

        pd.DataFrame({
            "Param": ["RISK_FREE", "MRP", "TAX", "TG"] + list(WEIGHTS.keys()),
            "Value": [RISK_FREE, MARKET_RISK_PREMIUM, GLOBAL_TAX_RATE, TERMINAL_GROWTH] + list(WEIGHTS.values())
        }).to_excel(writer, sheet_name="Assumptions", index=False)

        if args.backtest:
            bt = run_backtest(df["buy_signal"])
            bt.to_excel(writer, sheet_name="Backtest", index=False)


if __name__ == "__main__":
    main()
