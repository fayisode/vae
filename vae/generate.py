import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union
import numpy as np
import torch

import model as _m
import seeding as _se


# Configuration (moved to the top for easy modification)
IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image"))
FIGURE_SIZE = (8, 5)  # Consistent figure size
DPI = 300  # Set DPI for saving figures


def ensure_directory_exists(filepath: str) -> None:
    """Ensures that the directory for the given file path exists."""
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)


def save_plot(
    data: Union[List[float], np.ndarray],  # Type hint for data
    plot_type: str,
    filename: str,  # More descriptive name
    **kwargs,
) -> None:
    """Saves a plot to a file.

    Args:
        data: The data to plot.
        plot_type: The type of plot ("loss_curve", "loss_histogram", "image_grid").
        filename: The name of the file to save.
        **kwargs: Additional keyword arguments for specific plot types.
    """
    filepath = os.path.join(IMAGE_PATH, filename)
    ensure_directory_exists(filepath)

    fig = plt.figure(figsize=FIGURE_SIZE)

    if plot_type == "loss_curve":
        _plot_loss_curve(data, fig)
    elif plot_type == "loss_histogram":
        _plot_loss_histogram(data, fig)
    elif plot_type == "image_grid":
        _plot_image_grid(data, fig, **kwargs)
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")  # Handle invalid types

    fig.savefig(filepath, dpi=DPI)  # Save with specified DPI
    plt.close(fig)  # Close the figure to free memory


def _plot_loss_curve(data: Union[List[float], np.ndarray], fig: plt.FigureBase) -> None:
    """Plots a loss curve."""
    plt.plot(range(1, len(data) + 1), data, label="Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Gamma Training Loss Curve")
    plt.legend()
    plt.grid(True)


def _plot_loss_histogram(
    data: Union[List[float], np.ndarray], fig: plt.FigureBase
) -> None:
    """Plots a loss histogram."""
    plt.hist(data, bins=50, alpha=0.75, color="blue")
    plt.title("Loss Distribution Histogram")
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.grid(True)


def _plot_image_grid(
    data: Union[List[float], np.ndarray],
    fig: plt.FigureBase,
    grid_width: int,
    grid_height: int,
) -> None:
    """Plots a grid of images."""

    fig.subplots_adjust(hspace=0.15, wspace=0.10)
    for i in range(grid_width * grid_height):
        ax = fig.add_subplot(grid_height, grid_width, i + 1)
        ax.axis("off")
        ax.imshow(data[i, :, :, :])  # Directly use the data


def save_reconstructed_images(
    original_images: torch.Tensor,  # Type hint
    reconstructed_images: torch.Tensor,  # Type hint
    filename: str = "reconstructed_images.png",
    num_samples: int = 10,
) -> None:
    """Saves a grid of original and reconstructed images."""
    filepath = os.path.join(IMAGE_PATH, filename)
    ensure_directory_exists(filepath)

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 20))

    axes[0, 0].set_title("reconstruct", fontsize=14)
    axes[0, 1].set_title("original", fontsize=14)

    for i in range(num_samples):
        _plot_image(original_images[i][0], axes[i, 0])
        _plot_image(reconstructed_images[i][0], axes[i, 1])

    plt.tight_layout()
    fig.savefig(filepath, dpi=DPI)
    plt.close(fig)
    print(f"Reconstructed images saved to: {filepath}")


def _plot_image(image_tensor: torch.Tensor, ax: plt.Axes) -> None:
    """Helper function to plot a single image."""
    image = image_tensor.cpu().detach().numpy()
    ax.imshow(image, cmap="gray")
    ax.axis("off")


def compute_losses(
    kld_loss, reconst_loss, loss_function
) -> Dict[str, torch.Tensor]:  # Type hint beta_coef and return
    return {
        "kld": kld_loss,
        "reconst": reconst_loss,
        "loss_function": loss_function,
    }


def plot_loss_functions(
    data: Dict[str, List[float]], filename: str = "loss_functions_plot.png"
) -> None:
    """Plots the loss functions with improved visibility."""

    filepath = os.path.join(IMAGE_PATH, filename)
    ensure_directory_exists(filepath)

    plt.figure(figsize=FIGURE_SIZE)

    linestyles = ["-", "--", ":", "-."]  # Linestyles for differentiation
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]  # More distinct colors

    for i, (key, value) in enumerate(data.items()):
        if isinstance(value, list) and value:  # Check for empty lists
            plt.plot(
                range(1, len(value) + 1),
                value,
                label=key,
                alpha=0.8,  # Transparency
                linestyle=linestyles[i % len(linestyles)],  # Cycle through linestyles
                color=colors[i % len(colors)],  # Cycle through colors
            )

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss Values", fontsize=12)
    plt.title("Loss Function Components", fontsize=14)
    plt.legend(fontsize="medium", loc="best")  # Adjust legend size and location
    plt.grid(True)
    # plt.yscale("log")  # Log scale if appropriate
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI)
    plt.close()


def plot_loss_functions_grid(
    data: Dict[str, List[float]], filename: str = "loss_functions_plot_grid.png"
) -> None:
    """Plots loss functions in a grid layout."""

    filepath = os.path.join(IMAGE_PATH, filename)
    ensure_directory_exists(filepath)  # Make sure this function is defined

    num_plots = len(data)
    if num_plots == 0:
        return  # Nothing to plot

    # Calculate grid dimensions (adjust as needed)
    ncols = 2  # Number of columns
    nrows = (num_plots + ncols - 1) // ncols  # Calculate rows dynamically

    fig, axes = plt.subplots(
        nrows, ncols, figsize=FIGURE_SIZE, sharex=True
    )  # Share x-axis

    # Flatten axes if needed
    if num_plots == 1:  # Handle single plot case
        axes = [axes]
    elif nrows * ncols > num_plots:  # Handle cases where grid is larger than needed
        axes = axes.flatten()[:num_plots]
    else:
        axes = axes.flatten()

    plot_index = 0
    for key, value in data.items():
        if isinstance(value, list) and value:  # Check if list is not empty
            ax = axes[plot_index]
            ax.plot(range(1, len(value) + 1), value, label=key)
            ax.set_xlabel(
                "Epochs", fontsize=10
            )  # Smaller font size for individual plots
            ax.set_ylabel("Loss", fontsize=10)
            ax.set_title(key, fontsize=12)  # Title for each subplot
            ax.legend(fontsize=8)
            ax.grid(True)
            plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Loss Function Components", fontsize=14)  # Overall title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

    plt.savefig(filepath, dpi=DPI)
    plt.close()


def save_loss_plots(loss_histories: Dict[str, List[float]], name: str) -> None:
    """Saves individual loss plots and a combined loss function plot."""
    for loss_name, loss_values in loss_histories.items():
        save_plot(loss_values, "loss_curve", filename=f"{loss_name}_{name}.png")
    plot_loss_functions(loss_histories, f"{name}.png")  # Add .png extension
    plot_loss_functions_grid(loss_histories, f"{name}_grid.png")  # Add .png extension


def test_analysis(
    test_loader: DataLoader,
    model: _m.MyModel,
    device: torch.device,
) -> None:
    t_im, t_la = next(iter(test_loader))
    t_imc = t_im.to(device)
    t_re, t_mu_, t_logvar = model(t_imc)
    t_mu2, t_logvar2 = model.encode(t_imc)
    t_z = model.reparameterize(t_mu2, t_logvar2)
    t_re2 = model.decode(t_z)
    model.RE(t_re, t_imc) / len(t_re)
    save_reconstructed_images(t_re, t_im)


def analyze_clusters(
    test_loader: DataLoader,
    model: _m.MyModel,
    clusters: int,
    device: torch.device,
) -> None:
    x_1 = []
    y_1 = []
    for x, y in iter(test_loader):
        x_1.append(x)
        y_1.append(y)
    x_2 = torch.cat(x_1[:])
    y_2 = torch.cat(y_1[:])
    gt = y_2[:1000]
    model_prediction = model.predict(x_2[:1000].to(device))
    for i in range(0, clusters):
        print(
            "model prediction for ",
            i,
            " : ",
            model_prediction[gt == i]
            .reshape(1, len(model_prediction[gt == i]))
            .flatten(),
        )
        print(i, " : ", gt[gt == i].shape)


def plot_epoch_results(
    train_loss_arr: List[float],
    val_loss_arr: List[float],
    filename: str = "combined.png",
) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(train_loss_arr, label="Train Loss")
    plt.plot(val_loss_arr, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    fig.savefig(filename)
