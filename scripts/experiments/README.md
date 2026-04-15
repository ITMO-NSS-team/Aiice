
This section describes how to run experiments and update the AIICE benchmark leaderboard.

The repository provides multiple experimental setups for different models. Each model is defined by a separate configuration file located in the [`configs`](scripts/experiments/configs) directory.

---

To run an experiment, use the following command:

```bash
uv run scripts/experiments/cli.py \
    --config scripts/experiments/configs/baseline_mean.yaml
````

This will execute the selected model configuration and produce evaluation reports for the specified sea regions.

---

After an experiment is completed, you can add its results for a specific sea to the global leaderboard using:

```bash
uv run scripts/experiments/convert.py \
    --model baseline_mean \
    --sea "Kara Sea" \
    --report "outputs/baseline_mean/Kara Sea/report.yaml" \
    --csv .doc/leaderboard/table.csv \
    --html .doc/leaderboard/leaderboard.html \
    --readme README.md
```

This command:

* extracts metrics from the YAML report
* updates the central CSV leaderboard
* regenerates the HTML leaderboard for visualization
* inject the HTML leaderboard to the main README.md
