#! /bin/bash

python minecraft_probe_plot.py --results_dir minecraft_probe_results --metric "val_state_mean" --metric_title "Validation State Accuracy" --plot_file minecraft_probe_results_final_new_val_state_mean.png
python minecraft_probe_plot.py --results_dir minecraft_probe_results --metric "val_state_75p_mean" --metric_title "Validation State Accuracy (>75% state correct)" --plot_file minecraft_probe_results_final_new_val_state_75p.png
python minecraft_probe_plot.py --results_dir minecraft_probe_results --metric "val_state_90p_mean" --metric_title "Validation State Accuracy (>90% state correct)" --plot_file minecraft_probe_results_final_new_val_state_90p.png
python minecraft_probe_plot.py --results_dir minecraft_probe_results --metric "total_f1" --metric_title "Total F1 Score" --plot_file minecraft_probe_results_final_new_total_f1.png

python minecraft_probe_plot_attack.py --results_dir minecraft_probe_results_attack --metric "val_state_mean" --metric_title "Validation State Accuracy" --plot_file minecraft_attack_results_nui-32.png