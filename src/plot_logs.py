import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Fields to plot — edit this list to choose which columns show up
FIELDS = ["train_loss", "cover_loss", "secret_loss", "val_loss"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.path)

    plt.figure(figsize=(10, 6))

    for field in FIELDS:
        if field not in df.columns:
            print(f"Warning: '{field}' not found in CSV, skipping")
            continue
        sub = df[["step", field]].dropna()
        plt.plot(sub["step"], sub[field], label=field)

    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.title("Training metrics")
    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    plt.show()

if __name__ == "__main__":
    main()