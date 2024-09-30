import os
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# set the font size
plt.rc('font', size=16)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=16)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="minecraft_probe_results_final")
    parser.add_argument("--metric", type=str, default="val_state_mean")
    parser.add_argument("--metric_title", type=str, default="Validation State Accuracy")
    parser.add_argument("--plot_file", type=str, default="all.png")

    args = parser.parse_args()
    results_dir = args.results_dir
    metric = args.metric
    metric_title = args.metric_title
    plot_file = args.plot_file

    y_max = 100 if "f1" not in metric else 1

    # results_dir = "minecraft_probe_results_final"
    results_candidates = os.listdir(results_dir)
    # metric = "val_state_mean"
    # metric_title = "Validation State Accuracy"

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Load the results
    for candidate in results_candidates:
        if not candidate.endswith('.json'):
            continue
        candidate_path = os.path.join(results_dir, candidate)
        with open(candidate_path, 'r') as f:
            all_stats = json.load(f)

        curve_label = candidate.split("nui-")[1].split("_")[0]

        layers = list(all_stats.keys())
        # Increase the number of layers by 1 to account for the input layer
        layers = [str(int(layer) + 1) for layer in layers]
        
        # one point per layer
        ax.plot(
            layers,
            [layer_stats[metric] * y_max for layer_stats in all_stats.values()],
            marker="o",
            label=f"Num. of props: {curve_label}"
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_title)
    # ax.set_ylabel(">90% state correct accuracy")
    ax.set_ylim(0, y_max)
    # ax.set_xticks(list(all_stats.keys()))
    ax.set_xticks(layers)
    ax.grid()
    ax.set_title(f"{metric_title} vs Layer")
    ax.legend()
    # Sort the legend
    handles, labels = ax.get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0].split("Number of propositions : ")[1])))
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0].split("Num. of props: ")[1])))
    ax.legend(handles, labels)
    plt.savefig(os.path.join(results_dir, plot_file), bbox_inches="tight", transparent=True, dpi=300)
