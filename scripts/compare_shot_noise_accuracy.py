"""Descriptive accuracy and paired, exploratory sign tests for saved OO trials."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import scipy
from scipy.stats import binomtest

METHODS = ("seq", "newton_raw", "newton_stable")
LABELS = {"seq": "OO(seq)", "newton_raw": "Newton raw", "newton_stable": "Newton stabilized"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, precision=5):
    return "—" if value is None else f"{value:.{precision}f}"


def analyze(work):
    source = work / "summary.json"
    original = json.loads(source.read_text())
    manifest = json.loads((work / "manifest.json").read_text())
    geometry = f"LiH {manifest.get('bond_length_angstrom',1.6):g} Å"
    trial_count = len(manifest["seeds"])
    if trial_count < 2:
        raise ValueError("At least two trial seeds are required for a sample SD")
    runs = [row for row in original["runs"] if row["shots"] is not None]
    if any(row["status"] in ("pending", "running") for row in runs):
        raise ValueError("Accuracy comparison requires all finite-shot trajectories to finish")
    successful = [row for row in runs if row["completed"]]
    lookup = {(row["method"], row["shots"], row["seed"]): row["error_ha"] * 1000 for row in successful}
    if len(lookup) != len(successful):
        raise ValueError("Duplicate method/shot/seed results")
    descriptive, paired = [], []
    for shots in manifest["shots"]:
        for method in METHODS:
            seeds = [seed for seed in manifest["seeds"] if (method,shots,seed) in lookup]
            errors = np.array([lookup[method, shots, seed] for seed in seeds])
            descriptive.append({
                "shots": shots, "method": method, "n": len(errors), "requested": trial_count,
                "failed": trial_count-len(errors),
                "mean_mha": float(errors.mean()) if errors.size else None,
                "sample_sd_mha": float(errors.std(ddof=1)) if errors.size>1 else None,
                "median_mha": float(np.median(errors)) if errors.size else None,
                "rmse_mha": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
                "min_mha": float(errors.min()) if errors.size else None,
                "max_mha": float(errors.max()) if errors.size else None,
                "per_seed_errors_mha": {str(seed): float(value) for seed, value in zip(seeds, errors)},
            })
        for a, b in itertools.combinations(METHODS, 2):
            common_seeds = [seed for seed in manifest["seeds"] if (a,shots,seed) in lookup and (b,shots,seed) in lookup]
            differences = np.array([lookup[a, shots, seed] - lookup[b, shots, seed] for seed in common_seeds])
            wins, losses = int(np.sum(differences < 0)), int(np.sum(differences > 0))
            effective_n = wins + losses
            test = binomtest(wins, effective_n, p=0.5, alternative="two-sided") if effective_n else None
            interval = test.proportion_ci(confidence_level=0.95, method="exact") if test else None
            paired.append({
                "shots": shots, "a": a, "b": b, "n": len(differences), "requested": trial_count,
                "a_only_completed": sum((a,shots,seed) in lookup and (b,shots,seed) not in lookup for seed in manifest["seeds"]),
                "b_only_completed": sum((b,shots,seed) in lookup and (a,shots,seed) not in lookup for seed in manifest["seeds"]),
                "both_failed": sum((a,shots,seed) not in lookup and (b,shots,seed) not in lookup for seed in manifest["seeds"]),
                "a_wins": wins, "b_wins": losses,
                "ties": len(differences) - effective_n,
                "mean_a_minus_b_mha": float(differences.mean()) if differences.size else None,
                "median_a_minus_b_mha": float(np.median(differences)) if differences.size else None,
                "per_seed_a_minus_b_mha": {str(seed): float(value) for seed, value in zip(common_seeds, differences)},
                "sign_test_p_two_sided": float(test.pvalue) if test else 1.0,
                "a_win_probability_ci95_unadjusted": [float(interval.low), float(interval.high)] if interval else None,
            })
    # Holm step-down adjustment over all method pairs and shot levels (nine here).
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(paired)), key=lambda i: paired[i]["sign_test_p_two_sided"])):
        previous = max(previous, min(1.0, (len(paired) - rank) * paired[index]["sign_test_p_two_sided"]))
        paired[index]["holm_adjusted_p"] = previous
    result = {
        "geometry": geometry,
        "metric": "Final exact state energy minus FCI, in mHa; lower is more accurate",
        "pairing": "Same base trial seed within a shot level; trial seed windows are disjoint",
        "inference": "Exploratory two-sided paired sign tests; ties discarded; Holm adjustment over all comparisons",
        "alpha": 0.05, "descriptive": descriptive, "paired": paired,
        "provenance": {"summary_sha256": digest(source), "manifest_sha256": digest(work / "manifest.json"),
                       "script_sha256": digest(Path(__file__)), "python": sys.version,
                       "numpy": np.__version__, "scipy": scipy.__version__},
    }
    (work / "accuracy_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    archived = work / "accuracy_comparison_source.py"
    if Path(__file__).resolve() != archived:
        shutil.copy2(__file__, archived)
    plot_pairs(work, manifest, lookup)
    lines = [f"# Accuracy comparison: {geometry}", "",
             "Accuracy is the final exact state energy error from FCI. Sample SD describes trial-to-trial spread, not the uncertainty of one quantum measurement.",
             f"All results use {manifest['iterations']} macro iterations; every shot level has {trial_count} matched trial seeds per method.", "",
             "## Absolute accuracy", "",
             "Completed-only energy statistics and joint-completion paired tests must be read together with failure counts and endpoint accuracy rates from report.md.", "",
             "| Shots / word | Method | Completed | Mean ± sample SD (mHa) | Median (mHa) | RMSE (mHa) | Range (mHa) |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for row in descriptive:
        lines.append(f"| {row['shots']:,} | {LABELS[row['method']]} | {row['n']}/{row['requested']} | {number(row['mean_mha'])} ± {number(row['sample_sd_mha'])} | {number(row['median_mha'])} | {number(row['rmse_mha'])} | {number(row['min_mha'])}–{number(row['max_mha'])} |")
    lines += ["", "## Paired comparisons", "",
              "A win means a smaller exact error for A at the same seed, among jointly completed trajectories only. A negative mean difference favors A. Missing/failed trajectories are not assigned an artificial finite error.",
              "The sign test asks whether either method wins more often than probability 1/2 across independent trial seeds; it does not test equality of mean errors.", "",
              "| Shots / word | A vs B | Mean A−B (mHa) | A wins | Two-sided sign p | Holm-adjusted p |",
              "|---:|---|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['shots']:,} | {LABELS[row['a']]} vs {LABELS[row['b']]} | {number(row['mean_a_minus_b_mha'])} | {row['a_wins']}/{row['n']} | {row['sign_test_p_two_sided']:.4f} | {row['holm_adjusted_p']:.4f} |")
    lines += ["", "![Matched-seed errors](figures/paired_accuracy.png)", "", "## Interpretation", "",
              "- Descriptive accuracy can be compared now. General optimizer superiority requires more independent trials and conditions.",
              "- A non-rejection is not evidence of equivalence. The tests are exploratory and assess win probability rather than mean error."]
    smallest_p = 2.0 ** (1 - trial_count)
    lines.append(f"- With {trial_count} complete non-tied pairs, the smallest two-sided sign-test p-value is {smallest_p:.6g}. Fewer jointly completed pairs reduce the effective sample size.")
    if smallest_p > 0.05:
        lines.append("- These sign tests cannot reject at 5% with this sample size, even if every pair favors one method.")
    if 10000 in manifest["shots"]:
        ordered = sorted(lookup["seq",10000,seed] for seed in manifest["seeds"] if ("seq",10000,seed) in lookup)
        if len(ordered)>1:
            lines.append(f"- At 10,000 shots, the largest OO(seq) error is {ordered[-1]:.2f} mHa; the remaining {len(ordered)-1} errors range from {ordered[0]:.2f} to {ordered[-2]:.2f} mHa. Compare the mean and median to assess outlier sensitivity.")
    highest = max(manifest["shots"])
    comparison = next(row for row in paired if row["shots"]==highest and row["a"]=="seq" and row["b"]=="newton_raw")
    stable = next(row for row in paired if row["shots"]==highest and row["a"]=="seq" and row["b"]=="newton_stable")
    lines.append(f"- At {highest:,} shots, OO(seq) has a smaller error than raw Newton in {comparison['a_wins']}/{comparison['n']} jointly completed paired trials; mean OO(seq) minus raw Newton is {number(comparison['mean_a_minus_b_mha'])} mHa. Mean OO(seq) minus stabilized Newton is {number(stable['mean_a_minus_b_mha'])} mHa.")
    stable_groups = [row for row in original["summary"] if row["method"]=="newton_stable"]
    interventions = sum(row[key] for row in stable_groups for key in ("shifted_updates", "capped_updates", "backtracked_updates", "rejected_updates"))
    if interventions == 0:
        lines.append("- The original evaluation recorded no stabilization interventions. Differences between raw/stabilized numerical trajectories cannot be attributed to active Hessian shifts or step restrictions.")
    lines += ["", "Method reference: [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)."]
    (work / "accuracy_comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(paired, indent=2))
    print(f"Report: {work / 'accuracy_comparison.md'}")


def plot_pairs(work, manifest, lookup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(manifest["shots"]), figsize=(15, 4.5), sharey=True, squeeze=False)
    for ax, shots in zip(axes[0], manifest["shots"]):
        for number, seed in enumerate(manifest["seeds"], 1):
            values = [lookup.get((method, shots, seed), np.nan) for method in METHODS]
            ax.plot(range(len(METHODS)), values, marker="o", markersize=3 if len(manifest['seeds'])>8 else 5,
                    linewidth=.8 if len(manifest['seeds'])>8 else 1.2,
                    alpha=.4 if len(manifest['seeds'])>8 else .8, label=f"Trial {number}")
        ax.axhline(1.6, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(METHODS)), ["OO(seq)", "Newton\nraw", "Newton\nstabilized"])
        ax.set(title=f"{shots:,} shots / Pauli word", yscale="log")
        ax.grid(alpha=0.2, which="both")
    axes[0,0].set_ylabel("Final exact state energy error from FCI (mHa)")
    if len(manifest['seeds'])<=8:axes[0,-1].legend(fontsize=8)
    fig.suptitle(f"LiH {manifest.get('bond_length_angstrom',1.6):g} Å: matched-seed accuracy ({len(manifest['seeds'])} trial seeds)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(work / "figures" / f"paired_accuracy.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    analyze(parser.parse_args().work.resolve())
