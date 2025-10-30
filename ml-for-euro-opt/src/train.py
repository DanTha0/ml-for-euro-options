import pandas as pd
from pathlib import Path
from src.data_loader import process_each_file
from src.black_scholes import Option

RISK_FREE_RATE = 0.03

def main():
    process_each_file()

    processed = Path("data/processed")
    file = processed / "spx_eod_202301.csv"
    print(f"Using processed file: {file.name}")

    df = pd.read_csv(file)

    prices = []

    prices = []
    for _, row in df.iterrows():
        T = row["[dte]"] / 365.0
        sigma = row["[c_iv]"]
        if T <= 0 or sigma <= 0 or pd.isna(sigma):
            prices.append(0.0)
            continue

        opt = Option(
            S0=row["[underlying_last]"],
            K=row["[strike]"],
            T=T,
            r=RISK_FREE_RATE,
            sigma=sigma,
            o_type="call",
            q=0.0,
        )
        prices.append(opt.black_scholes())

    df["bs_price"] = prices

    out = processed / "black_scholes_results.csv"
    df.to_csv(out, index=False)
    print(f"Saved results to: {out}")

if __name__ == "__main__":
    main()
