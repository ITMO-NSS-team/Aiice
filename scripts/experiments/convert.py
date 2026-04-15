import argparse
import os

import pandas as pd
import yaml

METRIC_BETTER = {
    "mae": "min",
    "mse": "min",
    "rmse": "min",
    "psnr": "max",
    "ssim": "max",
    "iou": "max",
    "bin_accuracy": "max",
}


def load_report(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_mean_metrics(report: dict):
    rows = []

    for metric in METRIC_BETTER.keys():
        if metric in report:
            value = report[metric]["mean"]
            rows.append((metric, float(value)))

    return rows


def append_to_csv(csv_path, model, sea, rows):
    new_rows = [
        {
            "model": model,
            "sea": sea,
            "metric": metric,
            "value": value,
        }
        for metric, value in rows
    ]

    df_new = pd.DataFrame(new_rows)

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df = df.drop_duplicates(subset=["model", "sea", "metric"], keep="last")

    df.to_csv(csv_path, index=False)


def build_pivot(df: pd.DataFrame):
    return df.pivot_table(index=["sea", "metric"], columns="model", values="value")


def build_cell_styles(df: pd.DataFrame):
    pivot = df.pivot_table(
        index=["sea", "metric"], columns="model", values="value"
    ).sort_index()

    styles = []

    for row_idx, (idx, row) in enumerate(pivot.iterrows()):
        sea, metric = idx
        mode = METRIC_BETTER.get(metric, "max")

        values = row.astype(float)
        valid = values.notna()

        if not valid.any():
            continue

        best = values[valid].max() if mode == "max" else values[valid].min()

        for col_idx, col in enumerate(pivot.columns):
            val = values[col]

            if valid[col] and val == best:
                styles.append(
                    {
                        "selector": f"td.row{row_idx}.col{col_idx}",
                        "props": [("font-weight", "bold")],
                    }
                )

    return pivot, styles


def save_html(df: pd.DataFrame, html_path: str):
    pivot, styles = build_cell_styles(df)

    pivot.columns.name = None
    pivot.index.names = [None, None]

    styled = pivot.style.set_table_styles(styles)
    html = styled.to_html()

    with open(html_path, "w") as f:
        f.write(html)

    return html


def inject_into_readme(readme_path: str, html: str):
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    start = "<!-- benchmark -->"
    end = "<!-- benchmark -->"

    if start not in content or end not in content:
        raise ValueError("README must contain <!-- benchmark --> markers")

    before = content.split(start)[0]
    after = content.split(end)[-1]

    new_content = before + start + "\n" + html + "\n" + end + after

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--sea", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--html", required=True)
    parser.add_argument("--readme", required=True)

    args = parser.parse_args()

    report = load_report(args.report)
    rows = extract_mean_metrics(report)
    append_to_csv(args.csv, args.model, args.sea, rows)

    df = pd.read_csv(args.csv)

    html = save_html(df, args.html)
    inject_into_readme(args.readme, html)

    print(f"- updated CSV: {args.csv}")
    print(f"- generated HTML: {args.html}")


if __name__ == "__main__":
    main()
