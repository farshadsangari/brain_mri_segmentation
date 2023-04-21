import numpy as np
import matplotlib.pyplot as plt


def plot_metric(metric, metric_name, model_name, fig_size=(12, 5)):

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0, 0, 1, 1])
    epochs = len(metric[0])
    for label in ax.xaxis.get_ticklabels():
        label.set_color("black")
        label.set_rotation(0)
        label.set_fontsize(10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(round(epochs / 3)))
    ax.plot(
        [str(epoch) for epoch in range(1, epochs + 1)],
        metric[0],
        color="b",
        linewidth=2,
        label=f"{metric_name} Train",
    )
    ax.plot(
        [str(epoch) for epoch in range(1, epochs + 1)],
        metric[1],
        color="r",
        linewidth=2,
        label=f"{metric_name} Test",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("epoch", size=15)
    ax.set_ylabel(f"{metric_name}", size=15)
    ax.set_title(f"{metric_name} @ model : {model_name}")
    ax.legend(loc=0)
    plt.grid(alpha=0.3, zorder=0, linewidth=1)
    plt.legend(fontsize=12)
    plt.savefig("filename.png", dpi=300)


def plot_samples(
    model, myDataLoader, num_of_repeating_loader=3, batch_size=2, threshold=0
):
    num_of_rows = num_of_repeating_loader * batch_size
    cols_name = ["Original", "Mask", "Predicted"]
    fig, axes = plt.subplots(num_of_rows, 3, figsize=(20, 30))
    plt.subplots_adjust(wspace=0.05, hspace=0)
    for i in range(num_of_rows):
        for ax, col in zip(axes[i], cols_name):
            ax.set_title(col)
    image_index = 0
    for num_of_repeating_loader_ in range(num_of_repeating_loader):
        real_images, masks = next(iter(myDataLoader))
        real_images, masks = (
            real_images[:batch_size, :, :, :],
            masks[:batch_size, :, :, :],
        )
        model.to("cpu")
        predicteds = model(real_images)
        for batch_number in range(batch_size):
            real_image = (
                real_images[batch_number, :, :, :].detach().numpy().transpose((1, 2, 0))
            )
            real_image = np.clip(real_image, 0, 1)
            axes[image_index, 0].imshow(real_image[:, :, [2, 1, 0]])
            axes[image_index, 0].set_axis_off()
            mask = masks[batch_number, :, :, :].detach().numpy().transpose((1, 2, 0))
            axes[image_index, 1].imshow(mask)
            axes[image_index, 1].set_axis_off()
            predicted = (
                predicteds[batch_number, :, :, :].detach().numpy().transpose((1, 2, 0))
            )
            if threshold:
                predicted[predicted < threshold] = 0
                predicted[predicted > threshold] = 1
            axes[image_index, 2].imshow(predicted)
            axes[image_index, 2].set_axis_off()
            image_index += 1
    plt.show()


def Compare_Validations(
    list_of_metrics, models_name, metric_name, figsize=(10, 5), ylim=False
):

    fig, ax = plt.subplots(figsize=figsize)
    plt.rcdefaults()
    # Example data
    x_pos = np.arange(len(list_of_metrics))

    ax.bar(x_pos, list_of_metrics, align="center")
    ax.set_xticks(x_pos, models_name)
    ax.set_xlabel("Models", size=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel(f"{metric_name}", size=15)
    ax.set_title(f"Comapare Validation {metric_name} of Models")
    plt.grid(alpha=0.3, zorder=0, linewidth=1)
    plt.show()


def Plot_vs_Plot(
    list_of_metrics, list_of_metric_names, metric_name, fig_size=(12, 6), ylim=False
):

    fig = plt.figure(figsize=fig_size)
    epochs = len(list_of_metrics[0][0])
    ax = fig.add_axes([0, 0, 1, 1])
    for label in ax.xaxis.get_ticklabels():
        label.set_color("black")
        label.set_rotation(0)
        label.set_fontsize(10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(int(np.ceil((epochs / 2)))))

    colors = ["r", "g", "b", "c", "m", "y"]
    for index, metric in enumerate(list_of_metrics):
        ax.plot(
            [str(epoch) for epoch in range(epochs)],
            metric[0],
            color=colors[index],
            linewidth=2,
            label=f"{list_of_metric_names[index]} Train",
        )
        ax.plot(
            [str(epoch) for epoch in range(epochs)],
            metric[1],
            color=colors[index],
            linewidth=2,
            label=f"{list_of_metric_names[index]} Test",
        )
    ax.xaxis.set_major_locator(plt.MaxNLocator(round(epochs / 3)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("epoch", size=15)
    ax.set_ylabel(f"{metric_name}", size=15)
    ax.set_title(f"Compare {metric_name} of Models")
    ax.legend(loc=0)
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    plt.grid(alpha=0.3, zorder=0, linewidth=1)
    plt.legend(fontsize=12)
