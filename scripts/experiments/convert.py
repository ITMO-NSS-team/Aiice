import argparse
import glob
import os

import pandas as pd
import yaml

METRICS = [
    "mae",
    "rmse",
    "psnr",
    "ssim",
    "iou",
    "bin_accuracy",
]


def load_report(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_mean_metrics(report: dict):
    rows = []

    for metric in METRICS:
        if metric in report:
            value = report[metric]["mean"]
            rows.append((metric, float(value)))

    return rows


def find_yaml_in_dir(dir_path: str):
    """Find first .yaml or .yml file in directory."""
    for ext in ["*.yaml", "*.yml"]:
        matches = glob.glob(os.path.join(dir_path, ext))
        if matches:
            return matches[0]
    return None


def process_single_report(report_path, model, sea, forecast_len, step, csv_path):
    report = load_report(report_path)
    rows = extract_mean_metrics(report)
    if rows:
        append_to_csv(csv_path, model, sea, forecast_len, step, rows)
        print(f"  - {sea}: {len(rows)} metrics")
    else:
        print(f"  - {sea}: no metrics found")


def process_directory(dir_path, model, forecast_len, step, csv_path):
    """Walk subdirectories, treat each as a sea, find yaml inside."""
    for entry in sorted(os.listdir(dir_path)):
        sea_dir = os.path.join(dir_path, entry)
        if not os.path.isdir(sea_dir):
            continue

        yaml_path = find_yaml_in_dir(sea_dir)
        if yaml_path is None:
            print(f"  - {entry}: no yaml file found, skipping")
            continue

        process_single_report(yaml_path, model, entry, forecast_len, step, csv_path)


def append_to_csv(csv_path, model, sea, forecast_len, step, rows):
    new_rows = [
        {
            "model": model,
            "sea": sea,
            "metric": metric,
            "value": value,
            "forecast_len": forecast_len,
            "step": step,
        }
        for metric, value in rows
    ]

    df_new = pd.DataFrame(new_rows)

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df = df.drop_duplicates(
        subset=["model", "sea", "metric", "forecast_len", "step"],
        keep="last",
    )

    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--sea", default=None)
    parser.add_argument("--forecast_len", type=int, required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--csv", required=True)

    args = parser.parse_args()

    if args.sea is None:
        # --report is a directory containing subdirs named after seas
        if not os.path.isdir(args.report):
            print(
                f"Error: --sea not provided and --report is not a directory: {args.report}"
            )
            return

        print(f"Processing directory: {args.report}")
        process_directory(
            args.report, args.model, args.forecast_len, args.step, args.csv
        )

    else:
        # --sea provided, --report is a single yaml file
        if not os.path.isfile(args.report):
            print(f"Error: --report is not a file: {args.report}")
            return

        print(f"Processing single report for: {args.sea}")
        process_single_report(
            args.report, args.model, args.sea, args.forecast_len, args.step, args.csv
        )

    print(f"- updated CSV: {args.csv}")


if __name__ == "__main__":
    main()
