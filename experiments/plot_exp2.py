import json
import itertools
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib settings

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 10

line_styles_nv = {
    8: "-",
    16: "--",
    24: "-.",
    32: ":"
}
marker_styles = ["o", "v", "s", "P"]

line_colors_colorblind_nv = {
    8: "#D81B60",
    16: "#1E88E5",
    24: "#FFC107",
    32: "#4CAF50",
    40: "#FF5722",
    48: "#9C27B0",
    56: "#FF9800",
    64: "#795548"
}

results_dir = "/home/akhare/repos/tf_logic/_dump/distribution_shift_exps_final/"

configs = [
    {
        "nvs": [8, 16, 24],
        "ds": [96],
        "seeds": [101, 102, 103]
    }
]

probs_labels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
probs = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]
rules = [8, 16, 32, 64, 128, 256, 512]
rules_str = ["8", "16", "32", "64", "128", "256", "512"]

# need to create a 2D plot of the following:
# x-axis: prob
# y-axis: "TargetStatesAcc" in the json file
# each line is a different (n, d) value

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel("Probability")
ax.set_ylabel("Accuracy")
ax.set_ylim([0.0, 1.005])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.xaxis.set_tick_params(size=0)
ax.yaxis.set_tick_params(size=0)

for i, config in enumerate(configs):
    nvs = config["nvs"]
    ds = config["ds"]
    for n, d in itertools.product(nvs, ds):
        accs = {
            seed: []
            for seed in config["seeds"]
        }
        for j, p in enumerate(probs):
            for seed in config["seeds"]:
                file = f"gpt2_d{d}_L1_H1_seed{seed}__nv{n}_nr16-64_nsNone-None_ap0.20-0.30_bp0.20-0.30_len32768__nv{n}_nr16-64_nsNone-None_ap{p}-{p}_bp{p}-{p}_len8196.json"
                try:
                    with open(results_dir + file) as f:
                        data = json.load(f)
                    accs[seed].append(data["TargetStatesAcc"])
                except:
                    accs[seed].append(-1.0)

        # Group by seed and Plot the mean and fill with standard deviation
        print(accs)
        accs = np.array([accs[seed] for seed in config["seeds"]])
        mean = np.mean(accs, axis=0)
        std = np.std(accs, axis=0)
        ax.plot(probs_labels, mean, label=f"n={n}, d={d}", marker=marker_styles[i], markersize=3, linestyle=line_styles_nv[n], color=line_colors_colorblind_nv[n])
        ax.fill_between(probs_labels, mean - std, mean + std, alpha=0.2, color=line_colors_colorblind_nv[n])

# shade the area between x-ticks 0.2 and 0.3.
ax.axvspan('0.2', '0.3', alpha=0.4, color="#67CA67")

ax.legend(loc='lower right')
plt.show()
plt.savefig("plots/exp4_prob_plot.pdf", bbox_inches="tight")

# need to create a 2D plot of the following:
# x-axis: rules
# y-axis: "TargetStatesAcc" in the json file
# each line is a different (n, d) value
# each line is a different color

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel("Number of rules")
ax.set_ylabel("Accuracy")
ax.set_ylim([0.6, 1.001])
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"])

plt.style.use('seaborn-v0_8-colorblind')

for i, config in enumerate(configs):
    nvs = config["nvs"]
    ds = config["ds"]
    for nv, d in itertools.product(nvs, ds):
        accs = {
            seed: []
            for seed in config["seeds"]
        }
        for seed in config["seeds"]:
            for r in rules:
                file = f"gpt2_d{d}_L1_H1_seed{seed}__nv{nv}_nr16-64_nsNone-None_ap0.20-0.30_bp0.20-0.30_len32768__nv{nv}_nr{r}-{r}_nsNone-None_ap0.20-0.30_bp0.20-0.30_len8196.json"
                try:
                    with open(results_dir + file) as f:
                        data = json.load(f)
                    accs[seed].append(data["TargetStatesAcc"])
                except:
                    accs[seed].append(0.0)
        
        # Group by seed and Plot the mean and fill with standard deviation
        accs = np.array([accs[seed] for seed in config["seeds"]])
        mean = np.mean(accs, axis=0)
        std = np.std(accs, axis=0)
        ax.plot(rules_str, mean, label=f"n={nv}, d={d}", marker=marker_styles[i], markersize=3, linestyle=line_styles_nv[nv], color=line_colors_colorblind_nv[nv])
        ax.fill_between(rules_str, mean - std, mean + std, alpha=0.2, color=line_colors_colorblind_nv[nv])

# shade the area between 16 and 64.
ax.axvspan("16", "64", alpha=0.4, color="#67CA67")
ax.set_xticks(rules_str)
ax.xaxis.set_tick_params(size=0)
ax.yaxis.set_tick_params(size=0)

# Legend at the bottom right
ax.legend(loc='lower right')
plt.show()
plt.savefig("plots/exp4_rules_plot.pdf", bbox_inches="tight")



