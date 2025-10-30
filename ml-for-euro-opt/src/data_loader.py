import pandas as pd
from pathlib import Path

def process_each_file(raw_dir="data/raw", processed_dir="data/processed", sep=","):
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    files = list(raw_path.glob("*.txt"))

    for file in files:
        print(f"Processing {file.name}")
        df = pd.read_csv(
            file,
            sep=sep,
            engine="python",
            skipinitialspace=True,
            na_values=["", " "]
        )

        df.columns = [c.lower() for c in df.columns]

        out_file = processed_path / (file.stem + ".csv")
        df.to_csv(out_file, index=False)
        print(f"Saved processed file: {out_file}")

    print("All files processed successfully.")

