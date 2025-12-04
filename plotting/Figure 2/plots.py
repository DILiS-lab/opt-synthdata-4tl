import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_ode_fit(path_to_real_world,
                path_to_ode_baseline,
                attribute_to_plot,
                train_test_split,
                secondary_attribute=None,
                save_fig_path=None,
                xlabel_offset=0.03,
                linewidth=2,
                opacity=0.7,
                figsize=(6,6)):

    attributes = [attribute_to_plot]
    if secondary_attribute:
        attributes.append(secondary_attribute)
    fig, ax = plt.subplots(nrows=len(attributes), sharex=True, sharey=False, ncols=1, figsize=figsize)
    fig.supxlabel('Time', y=xlabel_offset, fontsize=10)

    true_file = os.path.join(os.getcwd(), path_to_real_world)
    true_data = pd.DataFrame(pd.read_csv(true_file))

    ode_fit = pd.read_csv(path_to_ode_baseline)
    if "algae" in ode_fit.columns and "rotifers" in ode_fit.columns:
        ode_fit["algae"] = ode_fit["algae"] / 1000000000
        ode_fit["rotifers"] = ode_fit["rotifers"] / 1000

    CUTOFF = int((1-train_test_split)*len(true_data))-1

    for plot_idx, attribute in enumerate(attributes):
        if len(attributes) == 1:
            ax.plot(range(0, CUTOFF+1), true_data[:CUTOFF+1][attribute], color="grey", linestyle="-", linewidth=1, alpha=1)
            ax.plot(range(0, CUTOFF+1),
                        ode_fit[:CUTOFF+1][attribute],
                        label="ODE baseline",
                        color='#51A537',
                        # marker=MARKER_LIST[2],
                        markersize=6,
                        linewidth=linewidth,
                        alpha=0.8,
                        linestyle="-")
            ax.set_ylabel(attribute)
            ax.set_xlim(0, CUTOFF)
        else:
            ax[plot_idx].plot(range(0, CUTOFF+1), true_data[:CUTOFF+1][attribute], color="grey", linestyle="-", linewidth=1, alpha=1)
            ax[plot_idx].plot(range(0, CUTOFF+1),
                        ode_fit[:CUTOFF+1][attribute],
                        label="ODE baseline",
                        color='#51A537',
                        # marker=MARKER_LIST[2],
                        markersize=6,
                        linewidth=linewidth,
                        alpha=0.8,
                        linestyle="-")
            ax[plot_idx].set_ylabel(str(attribute).capitalize())
            ax[plot_idx].set_xlim(0, CUTOFF)

    experiment = str(path_to_real_world).removesuffix(".csv")

    if save_fig_path:
        plt.savefig(save_fig_path+f"/{experiment}_ode-fit.png", format="PNG", dpi=300)
        plt.savefig(save_fig_path+f"/{experiment}_ode-fit.pdf", format="PDF", dpi=300)