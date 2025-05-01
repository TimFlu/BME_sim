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
ece_values = []  # in-memory store
all_realized_metrics = {'accuracy': [], 'auc': [], 'f1_score': [], 'recall': []}
all_estimated_metrics = {'accuracy': [], 'auc': [], 'f1_score': [], 'recall': []}
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
    TOTAL_SAMPLES = 1000
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
        fig, axs = plt.subplots(2, 1, figsize=(12, 18), layout='constrained', sharex=True)

        ax = axs[0]
        ax.hist(combined_probs[combined_labels == 1], bins=20, alpha=0.5, color='green', label='Label 1')
        ax.hist(combined_probs[combined_labels == 0], bins=20, alpha=0.5, color='m', label='Label 0')
        ax.vlines(x=0.5, ymin=0, ymax=np.max(np.histogram(combined_probs, bins=20)[0]), color='r', linewidth=2 ,linestyle='--', label='Threshold')
        ax.set_ylabel("Count")
        ax.set_xlabel("Confidence")
        # ax.text(0.5, 0.7, f"Ground Truth", fontsize=30, ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Confidence Histogram')
        ax.legend()

        # Realiability diagram
        ax = axs[1]
        num_bins = 20
        rbs = root_brier_score(combined_labels, combined_probs)
        

        bins = np.linspace(0, 1, num_bins + 1) # Bin edges
        bin_centers = (bins[:-1] + bins[1:]) / 2 # Bin centers

        bin_counts = np.zeros(num_bins) # Number of predictions in each bin
        bin_pos_label = np.zeros(num_bins) # Number of positive predictions in each bin
        bin_confidence = np.zeros(num_bins) # Mean confidence in each bin
        
        for label, pred_conf in zip(combined_labels, combined_probs):
            bin_idx = np.digitize(pred_conf, bins, right=True) - 1
            if bin_idx >= num_bins: # Account for edge case
                bin_idx = num_bins - 1
            bin_counts[bin_idx] += 1
            bin_confidence[bin_idx] += pred_conf
            bin_pos_label[bin_idx] += label

        bin_accuracy = np.nan_to_num(bin_pos_label / bin_counts)
        bin_confidence = np.nan_to_num(bin_confidence / bin_counts)
        ece = np.sum(bin_counts*np.abs(bin_accuracy - bin_confidence))/np.sum(bin_counts)
        ece_values.append(ece)

        ax.bar(bin_centers, bin_accuracy, width=1/num_bins, color='blue', label='Freq', alpha=0.7)
        for acc, diag in zip(bin_accuracy, bin_centers):
            if acc < diag:
                ax.bar(diag, diag-acc, bottom=acc, width=1/num_bins, color='red', alpha=0.2, hatch='/')
            else:
                ax.bar(diag, acc-diag, bottom=diag, width=1/num_bins, color='red', alpha=0.2, hatch='/')
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        ax.text(0.8, 0.05, f'ECE={ece:.3f}', bbox=dict(facecolor='grey', alpha=0.8, boxstyle='round', edgecolor='black'))
        # Text amount of samples in each bin
        # for i, txt in enumerate(bin_counts):
        #     plt.text(bin_centers[i], bin_accuracy[i]/2, f'{txt:.0f}', ha='center', va='bottom')
        ax.set_title(f"Reliability Diagram")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        ax.legend()


        output_path = os.path.join("static", "img", "generated.png")
        fig.savefig(output_path)
        plt.close()

  
    ## Column 4 Figure
    # Add point to accumulated list
    accumulated_points.append(slider_value/100)
    
    realized_metrics = calculate_metrics(combined_labels, combined_probs)
    estimated_metrics = calculate_CBPE_metrics(combined_probs)
    
    realized_metrics = {key: realized_metrics[key] for key in ['accuracy', 'auc', 'f1_score', 'recall']}
    estimated_metrics = {key: estimated_metrics[key] for key in ['accuracy', 'auc', 'f1_score', 'recall']}    

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
        fig, axs = plt.subplots(2, 1, figsize=(12, 18), layout='constrained', sharey=False)    
        # if axs is not np.ndarray:
        #     axs = [axs]

        
        axs[0].set_xlim(0, 1)
        axs[0].set_ylim(0, 1)
        axs[0].set_xlabel("Realized")
        axs[0].set_ylabel("Estimated")
        axs[0].plot(x, y, label="y = x", linestyle="--", color="gray")

        colors = ['blue', 'orange', 'green', 'red']
        for i, metric in enumerate(realized_metrics.keys()):
            axs[0].scatter(realized_metrics[metric], estimated_metrics[metric], s=700, c=colors[i], label=f"Realized {metric}", edgecolors='k', linewidths=3)
            axs[0].scatter(all_realized_metrics[metric], all_estimated_metrics[metric], s=700, c=colors[i], alpha=0.5)
        handles = [mlines.Line2D([], [], color='blue', marker='o', markersize=15, linestyle='None', label='Realized Accuracy'),
                    mlines.Line2D([], [], color='orange', marker='o', markersize=15, linestyle='None', label='Realized AUC'),
                    mlines.Line2D([], [], color='green', marker='o', markersize=15, linestyle='None', label='Realized F1 Score'),
                    mlines.Line2D([], [], color='red', marker='o', markersize=15,linestyle='None', label='Realized Recall')]
        labels = ['Accuracy', 'AUC', 'F1 Score', 'Recall']
        fig.legend(handles, labels, loc="upper left", ncols=1, bbox_to_anchor=(0.1, 1),
            columnspacing=1,  # Adjust the spacing between columns
            handlelength=2,  # Adjust the length of the legend handles
            # handleheight=2,  # Adjust the height of the legend handles
            frameon=False)


        axs[1].scatter(accumulated_points, ece_values, s=700, c='k', alpha=0.5)
        axs[1].scatter(slider_value/100, ece, s=700, c='k', edgecolors='r', linewidths=2)
        axs[1].set_xlim(-0.1, 1.1)
        axs[1].set_ylabel("ECE")
        axs[1].set_xlabel("I.D. Ratio")




        fig.savefig("static/img/accumulated.png")
        plt.close()


    return jsonify(success=True)

@app.route("/reset_accumulated", methods=["POST"])
def reset_accumulated():
    global accumulated_points
    accumulated_points = []
    global all_realized_metrics
    all_realized_metrics = {'accuracy': [], 'auc': [], 'f1_score': [], 'recall': []}
    global all_estimated_metrics
    all_estimated_metrics = {'accuracy': [], 'auc': [], 'f1_score': [], 'recall': []}
    global ece_values
    ece_values = []
    # Recreate base plot
    x = np.linspace(0, 10, 100)
    y = x
    with plt.style.context('/storage/homefs/tf24s166/code/BME_viz/data/plot_style.txt'):  # Use the custom style
        fig, axs = plt.subplots(2, 1, figsize=(12, 18), layout='constrained', sharey=False)    
        # if axs is not np.ndarray:
        #     axs = [axs]

        
        axs[0].set_xlim(0, 1)
        axs[0].set_ylim(0, 1)
        axs[0].set_xlabel("Realized")
        axs[0].set_ylabel("Estimated")
        x = np.linspace(0, 100, 100)
        y = x
        axs[0].plot(x, y, label="y = x", linestyle="--", color="gray")

        handles = [mlines.Line2D([], [], color='blue', marker='o', markersize=15, linestyle='None', label='Realized Accuracy'),
                    mlines.Line2D([], [], color='orange', marker='o', markersize=15, linestyle='None', label='Realized AUC'),
                    mlines.Line2D([], [], color='green', marker='o', markersize=15, linestyle='None', label='Realized F1 Score'),
                    mlines.Line2D([], [], color='red', marker='o', markersize=15,linestyle='None', label='Realized Recall')]
        labels = ['Accuracy', 'AUC', 'F1 Score', 'Recall']
        fig.legend(handles, labels, loc="upper left", ncols=1, bbox_to_anchor=(0.1, 1),
            columnspacing=1,  # Adjust the spacing between columns
            handlelength=2,  # Adjust the length of the legend handles
            # handleheight=2,  # Adjust the height of the legend handles
            frameon=False)

        axs[1].set_xlim(-0.1, 1.1)
        axs[1].set_ylabel("ECE")
        axs[1].set_xlabel("I.D. Ratio")

        fig.savefig("static/img/accumulated.png")
        plt.close()

    return jsonify(success=True)


if __name__ == "__main__":
    # initialize empty accumulated plot
    x = np.linspace(0, 10, 100)
    y = x
    with plt.style.context('/storage/homefs/tf24s166/code/BME_viz/data/plot_style.txt'):  # Use the custom style
        fig, axs = plt.subplots(2, 1, figsize=(12, 18), layout='constrained', sharey=False)    
        # if axs is not np.ndarray:
        #     axs = [axs]

        
        axs[0].set_xlim(0, 1)
        axs[0].set_ylim(0, 1)
        axs[0].set_xlabel("Realized")
        axs[0].set_ylabel("Estimated")
        x = np.linspace(0, 100, 100)
        y = x
        axs[0].plot(x, y, label="y = x", linestyle="--", color="gray")

        handles = [mlines.Line2D([], [], color='blue', marker='o', markersize=15, linestyle='None', label='Realized Accuracy'),
                    mlines.Line2D([], [], color='orange', marker='o', markersize=15, linestyle='None', label='Realized AUC'),
                    mlines.Line2D([], [], color='green', marker='o', markersize=15, linestyle='None', label='Realized F1 Score'),
                    mlines.Line2D([], [], color='red', marker='o', markersize=15,linestyle='None', label='Realized Recall')]
        labels = ['Accuracy', 'AUC', 'F1 Score', 'Recall']
        fig.legend(handles, labels, loc="upper left", ncols=1, bbox_to_anchor=(0.1, 1),
            columnspacing=1,  # Adjust the spacing between columns
            handlelength=2,  # Adjust the length of the legend handles
            # handleheight=2,  # Adjust the height of the legend handles
            frameon=False)

        axs[1].set_xlim(-0.1, 1.1)
        axs[1].set_ylabel("ECE")
        axs[1].set_xlabel("I.D. Ratio")

        fig.savefig("static/img/accumulated.png")
        plt.close()


    app.run(debug=True)
