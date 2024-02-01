import json
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np

# matplotlib settings
plt.rcParams.update({'font.size': 16})
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 12
plt.style.use('seaborn-v0_8-colorblind')

line_styles_nv = {
    8: "-",
    16: "--",
    24: "-.",
    32: ":"
}
marker_styles = ["o", "v", "s", "P"]

line_colors_colorblind_nv = {
    8: "tab:blue",
    16: "tab:orange",
    24: "tab:purple",
    32: "tab:red"
}

results_dir = "/home/akhare/repos/tf_logic/_dump/theory_attack_experiments_l1_comp/"

configs = [
    {
        "nvs": [8, 32],
        "ds": [96],
        "seeds": [102],
        "kappas": [10**i for i in range(-3, 10)]
    }
]

kappas_str = [f"{i}" for i in range(-3, 10)]

# need to create a 2D plot of the following:
# x-axis: lambda
# y-axis: "TargetStatesAcc" in the json file
# each line is a different (n, d) value

fig = plt.figure(figsize=(8, 5))
ax = plt.axes()
ax.set_xlabel(r"$\kappa$ ($\log_{10}$)")
ax.set_ylabel("Target Match Accuracy")
ax.set_ylim([0.0, 1.01])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

for i, config in enumerate(configs):
    nvs = config["nvs"]
    ds = config["ds"]
    for n, d in itertools.product(nvs, ds):
        accs = {
            seed: []
            for seed in config["seeds"]
        }
        for seed in config["seeds"]:
            file = f"gpt2_nv{n}_ns1_nr16-64_ap0.20-0.30_bp0.20-0.30_cl2-5_ntr32768_ntt8196_seed{seed}.json"
            try:
                with open(results_dir + file) as f:
                    data = json.load(f)
                for kappa in config["kappas"]:
                    accs[seed].append(data[str(kappa)]["TargetStatesAcc"])
            except Exception as e:
                print(e)
                accs[seed].append(-1.0)

        # Group by seed and Plot the mean and fill with standard deviation
        print(accs)
        accs = np.array([accs[seed] for seed in config["seeds"]])
        mean = np.mean(accs, axis=0)
        std = np.std(accs, axis=0)
        ax.plot(kappas_str, mean, label=f"n={n}, d={d}", marker=marker_styles[i], markersize=3, linestyle=line_styles_nv[n], color=line_colors_colorblind_nv[n])
        ax.fill_between(kappas_str, mean - std, mean + std, alpha=0.2, color=line_colors_colorblind_nv[n])

ax.legend(loc='lower left')
plt.show()
plt.savefig("plots/exp_theory_comp_plot.png", bbox_inches="tight")