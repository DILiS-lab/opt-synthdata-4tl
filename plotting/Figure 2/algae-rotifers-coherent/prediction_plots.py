import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np
import csv

# Define your font
fpath = Path(mpl.get_data_path(), "fonts/ttf/cmr10.ttf")
fontprop = fm.FontProperties(fname=fpath, size=14)
fontprop_small = fm.FontProperties(fname=fpath, size=14)

sns.set_style("whitegrid")

COLORS=["#888888", "#FAA719", "#00799D", "#44AA99", "#CC6677"]
MARKER_LIST = ['o', '^', 's', 'P', 'P']

def plot_run_comparison(folder_path_transfer_learning,
                        path_to_real_world,
                        path_to_baseline,
                        attribute_to_plot,
                        batch_number,
                        models,
                        train_test_split,
                        input_attributes=None,
                        save_fig_path=None,
                        ylim=(None, None),
                        xlim=(None, None),
                        text_offset=(0,0),
                        linewidth=1):
    for model in models:
        plt.figure(figsize=(7, 4), dpi=200)

        true_file = os.path.join(get_root(), path_to_real_world)

        x_test_file = os.path.join(os.getcwd(), folder_path_transfer_learning, f"X_test.npy")
        y_true_file = os.path.join(os.getcwd(), folder_path_transfer_learning, f"y_true.npy")

        file = open(os.path.join(os.getcwd(), folder_path_transfer_learning, "features.csv"), "r")
        attribute_idx = list(csv.reader(file))[0].index(attribute_to_plot)
        file.close()

        if input_attributes:
            input_attributes_data = pd.DataFrame(pd.read_csv(true_file)[input_attributes]).astype(int)

        true_data = pd.DataFrame(pd.read_csv(true_file)[attribute_to_plot]).astype(int)
        x_test_data = np.load(x_test_file)[batch_number, :, attribute_idx].reshape(-1, 1)
        y_true_data = np.concatenate((x_test_data[-1], np.load(y_true_file)[batch_number, :, attribute_idx])).reshape(-1, 1)

        start_index = find_overlap_index(true_data, x_test_data)
        
        if input_attributes:
            # Plot other input attributes
            plt.plot(range(0, start_index+1), input_attributes_data[:start_index+1], color='lightgrey', linestyle=":", label=input_attributes)
            plt.plot(range(start_index, len(x_test_data) + start_index), input_attributes_data[start_index:start_index+len(x_test_data)], color=COLORS[0], linestyle=':', label=f"input {input_attributes}", linewidth=linewidth)

        # Plot input attributes that is also being predicted
        plt.plot(range(0, start_index+1), true_data[:start_index+1], color='lightgrey', linestyle="--", label="new infections")
        plt.plot(range(start_index+len(x_test_data)+len(y_true_data)-2, len(true_data)), true_data[start_index+len(x_test_data)+len(y_true_data)-2:], color='lightgrey', linestyle="--")
        plt.plot(range(start_index, len(x_test_data) + start_index), x_test_data, color=COLORS[0], linestyle='--', label=f"input", linewidth=linewidth)
        plt.plot(range(len(x_test_data)-1 + start_index,
                        len(x_test_data) + len(y_true_data)-1 + start_index),
                        y_true_data,
                        color=COLORS[0],
                        label="ground truth",
                        linewidth=linewidth)
        
        # Add vertical line to show training cut-off
        plt.axvline(x = int((1-train_test_split)*len(true_data))-1, color = 'black', linestyle = '-', linewidth = '1')
        plt.text(int((1-train_test_split)*len(true_data))-text_offset[0], text_offset[1], 'train cutoff 2', rotation=90, color = 'black', fontproperties=fontprop_small)

        # Print Transfer Learning Prediction
        y_pred_file = f"{folder_path_transfer_learning}/{model}-y_pred.npy"
        y_pred_data = np.load(y_pred_file)[batch_number, :, attribute_idx].reshape(-1, 1)
        y_pred_data = np.concatenate((x_test_data[-1].reshape(-1, 1), y_pred_data))
        plt.plot(range(len(x_test_data)-1+start_index, len(x_test_data)+len(y_true_data)-1+start_index),
                    y_pred_data,
                    label="TL run",
                    color=COLORS[1],
                    marker=MARKER_LIST[1],
                    markersize=6,
                    linewidth=linewidth)
        
        # Print Baseline Prediction
        y_pred_file = f"{path_to_baseline}/{model}-y_pred.npy"
        y_pred_data = np.load(y_pred_file)[batch_number, :, attribute_idx].reshape(-1, 1)
        y_pred_data = np.concatenate((x_test_data[-1].reshape(-1, 1), y_pred_data))
        plt.plot(range(len(x_test_data)-1+start_index, len(x_test_data) + len(y_true_data)-1+start_index),
                    y_pred_data,
                    label="basic ML run",
                    color=COLORS[2],
                    marker=MARKER_LIST[2],
                    markersize=6,
                    linewidth=linewidth)
        plt.legend(fancybox=True, fontsize=18, prop=fontprop)
        plt.xlabel('Time', fontproperties=fontprop)
        plt.ylabel('Number of new infections', fontproperties=fontprop)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks(fontproperties=fontprop)
        plt.yticks(fontproperties=fontprop)
        model = model.replace("PyTorch_Lightning_", "").replace("_", " ").replace("Custom ", "")
        #plt.title(f"Comparison of Transfer Learning and Basic ML Run for {model}", pad=10, fontproperties=fontprop)
        plt.tight_layout()
        if save_fig_path:
            plt.savefig(save_fig_path, format="pdf")
        else:
            plt.show()


def plot_model_comparison(path_to_transfer_learning, path_to_real_world, attribute_to_plot, batch_number, models, title, train_test_split, xlim=(None, None), ylim=(None, None), cutoff_label_offset=(0,0), figsize=(10,4)):
    true_file = os.path.join(get_root(), path_to_real_world)

    x_test_file = os.path.join(os.getcwd(), path_to_transfer_learning, f"X_test.npy")
    y_true_file = os.path.join(os.getcwd(), path_to_transfer_learning, f"y_true.npy")

    file = open(os.path.join(os.getcwd(), path_to_transfer_learning, "features.csv"), "r")
    attribute_idx = list(csv.reader(file))[0].index(attribute_to_plot)
    file.close()

    true_data = pd.DataFrame(pd.read_csv(true_file)[attribute_to_plot]).astype(int)
    x_test_data = np.load(x_test_file)[batch_number, :, attribute_idx].reshape(-1, 1)
    y_true_data = np.load(y_true_file)[batch_number, :, attribute_idx].reshape(-1, 1)

    start_index = find_overlap_index(true_data, x_test_data)
    
    plt.figure(figsize=figsize, dpi=200)
    plt.plot(range(0, len(true_data)), true_data, color='lightgrey', linestyle=":")
    plt.plot(range(start_index, len(x_test_data)+start_index), x_test_data, color=COLORS[0], linestyle='--', label="Input", linewidth=2)
    plt.plot(range(len(x_test_data)+start_index, len(x_test_data)+len(y_true_data)+start_index), y_true_data, color=COLORS[0], label="Ground Truth", linewidth=2)

    # Add vertical line to show training cut-off
    plt.axvline(x = int((1-train_test_split)*len(true_data)-1), color = 'black', linestyle = '-', linewidth = '1.0')
    plt.text(int(int((1-train_test_split)*len(true_data)-1))-cutoff_label_offset[0], cutoff_label_offset[1], 'Train Cutoff', rotation=90, color = 'black', fontproperties=fontprop_small)

    for model in models:
        y_pred_file = f"{path_to_transfer_learning}/{model}-y_pred.npy"
        y_pred_data = np.load(y_pred_file)[batch_number, :, attribute_idx].reshape(-1, 1)
        y_pred_data = np.concatenate((x_test_data[-1].reshape(-1, 1), y_pred_data))
        plt.plot(range(len(x_test_data)-1+start_index, len(x_test_data)+len(y_true_data)+start_index),
                    y_pred_data,
                    label=model.replace("PyTorch_Lightning_", "").replace("_", " ").replace("Custom ", ""),
                    color=COLORS[models.index(model)+1],
                    marker=MARKER_LIST[models.index(model)+1],
                    markersize=4,
                    linewidth=2)
    
    plt.legend(fancybox=True, ncol=1, prop=fontprop_small)
    plt.xlabel('Timesteps', fontproperties=fontprop)
    plt.ylabel('Value', fontproperties=fontprop)
    plt.xticks(fontproperties=fontprop)
    plt.yticks(fontproperties=fontprop)
    if not xlim == (None, None):
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(0, len(true_data))
    plt.ylim(ylim[0], ylim[1])
    plt.title(title, fontproperties=fontprop)
    plt.show()


def find_overlap_index(all, subset):
    all = all.to_numpy()
    subset = subset[:5]
    for i in range(len(all)-len(subset)+1):
        if (all[i:len(subset)+i] == subset).all():
            return i
    raise ValueError("No overlap found")

def get_root():
    # load context data
    path = os.path.abspath('')
    path = os.path.normpath(path)
    components = path.split(os.path.sep)
    src_index = components.index("src")
    components = components[:src_index]
    return os.path.sep.join(components)