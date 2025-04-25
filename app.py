from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
import numpy as np
import os
import pickle as pkl
import pandas as pd
import sys
sys.path.append('/storage/homefs/tf24s166/code/BME_viz/') 

from data.utils import *

accumulated_points = []  # in-memory store
all_realized_metrics = {'accuracy': [], 'bal_accuracy': [], 'f1_score': [], 'recall': []}
all_estimated_metrics = {'accuracy': [], 'bal_accuracy': [], 'f1_score': [], 'recall': []}
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    # Get the slider value from the request
    data = request.json
    slider_value = data.get("slider_value", 50)


    # Get cheXpert data for I.D. and NIH data for ood
    PATH_TO_OUTS = '/storage/homefs/tf24s166/code/BME_viz/data/uncal_scores_labs.pkl'

    with open(PATH_TO_OUTS, 'rb') as f:
        uncal_scores_labs = pkl.load(f)

    # All data for model trained on cheXpert with pathology Pleural Effusion
    chexpert_pleural_eff_data= uncal_scores_labs['Pleural Effusion_chestxpert']['seed_1']

    # Load ID data
    id_test_conf = chexpert_pleural_eff_data['id_out_test']
    id_test_labels = chexpert_pleural_eff_data['id_out_labs']

    # load NIH data
    nih_test_conf = chexpert_pleural_eff_data['ood1_out']
    nih_test_labels = chexpert_pleural_eff_data['ood1_labs']

    # Resample cheXpert and NIH such that their ratio matches the slider value
    TOTAL_SAMPLES = 5000
    chexpert_samples = int((slider_value / 100) * TOTAL_SAMPLES)
    nih_samples = int(TOTAL_SAMPLES - chexpert_samples)
    
    # Resample cheXpert data
    chexpert_indices = np.random.choice(len(id_test_conf), chexpert_samples, replace=False)

    chexpert_labels = id_test_labels[chexpert_indices]
    chexpert_probs = id_test_conf[chexpert_indices]

    # Resample NIH data
    nih_indices = np.random.choice(len(nih_test_conf), nih_samples, replace=False)
    nih_labels = nih_test_labels[nih_indices]
    nih_probs = nih_test_conf[nih_indices]

    # Combine the two datasets
    combined_labels = np.concatenate((chexpert_labels, nih_labels))
    combined_probs = np.concatenate((chexpert_probs, nih_probs))

    with plt.style.context('/storage/homefs/tf24s166/code/BME_viz/data/plot_style.txt'):  # Use the custom style
        fig, axs = plt.subplots(2, 1, figsize=(12, 15), layout='constrained', sharey=True)
      

        # Get histogram data
        bins = np.linspace(0, 1, 21)
        bin_width = bins[1] - bins[0]

        counts, bin_edges = np.histogram(combined_probs, bins=bins)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_lefts = bin_edges[:-1]

        ax = axs[0]
        # Plot split-color bars
        for left, count in zip(bin_lefts, counts):
            color_ = left
            if left < 0.5:
                color_ = 1 - left
            green_height = count * color_
            red_height = count - green_height
            ax.bar(left+0.5*bin_width, green_height, width=bin_width, color='green', alpha=0.5)
            ax.bar(left+0.5*bin_width, red_height, width=bin_width, bottom=green_height, color='red', alpha=0.5)
        ax.text(0.5, 0.7, 'Estimated Correct and Wrong Predictions', fontsize=30, ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel("Count")
        ax.set_xlabel("Confidence")


        ax = axs[1]
        # ax.hist(combined_probs[combined_labels == 1], bins=20, alpha=0.5, label='Combined Probabilities')
        # ax.hist(combined_probs[combined_labels == 0], bins=20, alpha=0.5, label='Combined Probabilities')
        # # ax.hist(combined_probs, bins=20, alpha=0.5, label='Combined Probabilities')
        # Plot split-color bars
        bins = np.linspace(0, 1, 21)
        bin_width = bins[1] - bins[0] 

        gt_tp, gt_tp_edges = np.histogram(combined_probs[combined_labels == 1], bins=bins)
        gt_tn, gt_tn_edges = np.histogram(combined_probs[combined_labels == 0], bins=bins)

        true_preds = np.concatenate([gt_tn[:9], gt_tp[9:]])
        total_counts_per_bin = np.histogram(combined_probs, bins=bins)[0]
        # print(np.sum((true_preds))/np.sum(total_counts_per_bin))
        # print(calculate_metrics(combined_labels, combined_probs)['accuracy'])

        bin_lefts = gt_tp_edges[:-1]
        for left, count, total_counts in zip(bin_lefts, true_preds, total_counts_per_bin):
            green_height = count
            red_height = total_counts - green_height
            ax.bar(left+0.5*bin_width, green_height, width=bin_width, color='green', alpha=0.5)
            ax.bar(left+0.5*bin_width, red_height, width=bin_width, bottom=green_height, color='red', alpha=0.5)
        ax.set_ylabel("Count")
        ax.set_xlabel("Confidence")
        ax.text(0.5, 0.7, f"Ground Truth", fontsize=30, ha='center', va='center', transform=ax.transAxes)

        output_path = os.path.join("static", "img", "generated.png")
        fig.savefig(output_path)
        plt.close()

  
    ## Column 4 Figure
    # Add point to accumulated list
    accumulated_points.append(slider_value/100)
    
    realized_metrics = calculate_metrics(combined_labels, combined_probs)
    estimated_metrics = calculate_CBPE_metrics(combined_probs)
    
    realized_metrics = {key: realized_metrics[key] for key in ['accuracy', 'bal_accuracy', 'f1_score', 'recall']}
    estimated_metrics = {key: estimated_metrics[key] for key in ['accuracy', 'bal_accuracy', 'f1_score', 'recall']}    

    # Store the metrics in the global list
    for metric in realized_metrics.keys():
        if metric in all_realized_metrics:
            all_realized_metrics[metric].append(realized_metrics[metric])
            all_estimated_metrics[metric].append(estimated_metrics[metric])
        else:
            continue

    # Generate accumulated output plot
    x = np.linspace(0, 100, 100)
    y = x
    with plt.style.context('/storage/homefs/tf24s166/code/BME_viz/data/plot_style.txt'):  # Use the custom style
        fig, axs = plt.subplots(1, 1, figsize=(10, 10), layout='constrained', sharey=True)    
        if axs is not np.ndarray:
            axs = [axs]

        for ax in axs:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Realized")
            ax.set_ylabel("Estimated")
            ax.plot(x, y, label="y = x", linestyle="--", color="gray")

        colors = ['blue', 'orange', 'green', 'red']
        for i, metric in enumerate(realized_metrics.keys()):
            axs[0].scatter(realized_metrics[metric], estimated_metrics[metric], s=700, c=colors[i], label=f"Realized {metric}")
            axs[0].scatter(all_realized_metrics[metric], all_estimated_metrics[metric], s=700, c=colors[i], alpha=0.5)
        handles = [mlines.Line2D([], [], color='blue', marker='o', markersize=15, linestyle='None', label='Realized Accuracy'),
                    mlines.Line2D([], [], color='orange', marker='o', markersize=15, linestyle='None', label='Realized Bal Accuracy'),
                    mlines.Line2D([], [], color='green', marker='o', markersize=15, linestyle='None', label='Realized F1 Score'),
                    mlines.Line2D([], [], color='red', marker='o', markersize=15,linestyle='None', label='Realized Recall')]
        labels = ['Accuracy', 'Bal Accuracy', 'F1 Score', 'Recall']
        fig.legend(handles, labels, loc="upper left", ncols=1, bbox_to_anchor=(0.1, 1),
            columnspacing=1,  # Adjust the spacing between columns
            handlelength=2,  # Adjust the length of the legend handles
            # handleheight=2,  # Adjust the height of the legend handles
            frameon=False)

        fig.savefig("static/img/accumulated.png")
        plt.close()


    return jsonify(success=True)

@app.route("/reset_accumulated", methods=["POST"])
def reset_accumulated():
    global accumulated_points
    accumulated_points = []
    global all_realized_metrics
    all_realized_metrics = {'accuracy': [], 'bal_accuracy': [], 'f1_score': [], 'recall': []}
    global all_estimated_metrics
    all_estimated_metrics = {'accuracy': [], 'bal_accuracy': [], 'f1_score': [], 'recall': []}
    # Recreate base plot
    x = np.linspace(0, 10, 100)
    y = x
    with plt.style.context('/storage/homefs/tf24s166/code/BME_viz/data/plot_style.txt'):  # Use the custom style
        fig, axs = plt.subplots(1, 1, figsize=(10, 10), layout='constrained', sharey=True)    
        if axs is not np.ndarray:
            axs = [axs]

        for ax in axs:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Realized")
            ax.set_ylabel("Estimated")
            ax.plot(x, y, label="y = x", linestyle="--", color="gray")

        fig.savefig("static/img/accumulated.png")
        plt.close()

    return jsonify(success=True)


if __name__ == "__main__":
    # initialize empty accumulated plot
    plt.figure()
    x = np.linspace(0, 10, 100)
    y = x
    with plt.style.context('/storage/homefs/tf24s166/code/BME_viz/data/plot_style.txt'):  # Use the custom style
        fig, axs = plt.subplots(1, 1, figsize=(10, 10), layout='constrained', sharey=True)    
        if axs is not np.ndarray:
            axs = [axs]

        for ax in axs:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Realized")
            ax.set_ylabel("Estimated")
            ax.plot(x, y, label="y = x", linestyle="--", color="gray")

        fig.savefig("static/img/accumulated.png")
        plt.close()


    app.run(debug=True)
