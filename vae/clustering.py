import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # Or use UMAP
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import model as _m

# Assuming 'generate' is a custom module with helper functions
import generate  # Make sure this import is correct


def gamma_training(
    model: _m.MyModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    steps: int,
    gamma: float,
) -> Dict[str, List[float]]:
    """Initial gamma training for a specified number of steps.

    Args:
        model: The VAE model.
        dataloader: The data loader.
        optimizer: The optimizer.
        steps: Number of training steps.
        gamma: Initial gamma value.

    Returns:
        A dictionary containing loss histories.
    """

    print(f"Starting initial gamma training for {steps} steps.")
    loss_histories: Dict[str, List[float]] = {  # Type hint the dictionary
        "kld": [],
        "mse": [],
        "reconst": [],
        "loss_function": [],
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Get device outside the loop

    for batch_idx, (data, _) in tqdm(enumerate(dataloader)):
        data = data.to(device)  # Use .to(device) instead of Variable and cuda()

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = model.RE(recon_batch, data) + gamma * model.KLD(mu, logvar)
        loss.backward()

        n_iter = len(dataloader)
        L = generate_gamma_schedule(
            n_iter, gamma
        )  # Ensure generate_gamma_schedule is defined

        losses = generate.compute_losses(
            model, recon_batch, data, mu, logvar, L[batch_idx]
        )  # Ensure compute_losses is defined
        optimizer.step()

        # Log losses
        for key, value in losses.items():
            loss_histories[key].append(value.item())

        if batch_idx + 1 == steps:
            print("Training completed")
            break

    generate.save_loss_plots(
        loss_histories, "gamma_loss"
    )  # Ensure save_loss_plots is defined
    return loss_histories


def generate_gamma_schedule(n_iter: int, gamma: float) -> np.ndarray:
    """Generates a gamma schedule.

    Args:
        n_iter: Total number of iterations.
        gamma: Initial gamma value.

    Returns:
        A numpy array representing the gamma schedule.
    """
    n_cycle = 5
    ratio = 0.5
    stop = gamma + 1
    schedule = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - gamma) / (period * ratio)

    for c in range(n_cycle):
        v, i = gamma, 0
        while v <= stop and (int(i + c * period) < n_iter):
            schedule[int(i + c * period)] = v
            v += step
            i += 1

    return schedule


def vae_clustering_pretrained(
    model: _m.MyModel,
    data_loader: DataLoader,
    batch_size: int,
    n_clusters: int,
    train_set: torch.utils.data.Dataset,
) -> Tuple[
    GaussianMixture,
    np.ndarray,
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """Performs clustering using a pretrained VAE.

    Args:
        model: The pretrained VAE model.
        data_loader: The data loader.
        batch_size: Batch size.
        n_clusters: Number of clusters.
        train_set: The training dataset.

    Returns:
        A tuple containing the fitted GaussianMixture model and cluster predictions.
    """

    k = round(len(data_loader) / 2)
    subset = np.random.randint(len(data_loader), size=k * batch_size)
    train_subset_loader = DataLoader(
        torch.utils.data.Subset(train_set, subset), batch_size=batch_size
    )  # Add batch_size

    Z: List[torch.Tensor] = []  # Type hint the list
    Y: List[torch.Tensor] = []  # Type hint the list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"GMM Initializing with {k * batch_size} MC samples")
    with torch.no_grad():
        for data, y in tqdm(
            train_subset_loader
        ):  # No need for enumerate if you don't use batch_idx
            data = data.to(device)

            mu, _ = model.encode(data)  # No need to store logvar if not used
            Z.append(mu)
            Y.append(y)

    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).detach().cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(Z)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        max_iter=int(1e04),
        means_init=kmeans.cluster_centers_,
        reg_covar=1e-4,
        verbose=0,  # Set to 0 to suppress output during fitting (or 2 for detailed output)
        random_state=100,
    )

    pre = gmm.fit_predict(Z)

    model.pi_.data = torch.tensor(gmm.weights_, device=device).float()
    model.mu_c.data = torch.tensor(gmm.means_, device=device).float()
    model.log_var_c.data = torch.log(
        torch.tensor(gmm.covariances_, device=device).float()
    )

    return gmm, pre, Z, Y


def visualize_and_evaluate_clustering(
    model: _m.MyModel,
    data_loader: torch.utils.data.DataLoader,
    batch_size: int,
    n_clusters: int,
    train_set: torch.utils.data.Dataset,
    has_true_labels: bool = True,  # Flag if you have true labels
) -> Tuple[GaussianMixture, np.ndarray, float, float]:  # Add return types
    """Performs clustering, visualizes results, and evaluates performance.

    Args:
        model: The pretrained VAE model.
        data_loader: The data loader.
        batch_size: Batch size.
        n_clusters: Number of clusters.
        train_set: The training dataset.
        has_true_labels: Whether the dataset has true labels (Y).

    Returns:
        A tuple containing the fitted GaussianMixture model, cluster predictions, NMI, and ARI.
    """

    gmm, pre, Z, Y = vae_clustering_pretrained(
        model, data_loader, batch_size, n_clusters, train_set
    )
    visualize_latent_space(Z, pre, Y if has_true_labels else None)

    nmi, ari = calculate_clustering_metrics(Y, pre) if has_true_labels else (None, None)
    print(f"NMI: {nmi}")
    print(f"ARI: {ari}")

    return gmm, pre, nmi, ari  # Return the metrics


def visualize_latent_space(Z: np.ndarray, pre: np.ndarray, Y: np.ndarray = None):
    """Visualizes the latent space using t-SNE."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)  # Adjust perplexity
    Z_embedded = tsne.fit_transform(Z)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z_embedded[:, 0], Z_embedded[:, 1], c=pre, label="Predicted Clusters")
    if Y is not None:
        plt.scatter(
            Z_embedded[:, 0],
            Z_embedded[:, 1],
            c=Y,
            marker="x",
            label="True Labels",
            alpha=0.5,
        )  # Overlay true labels
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()  # Show legend if both predicted and true are plotted.
    plt.colorbar(label="Cluster")
    plt.savefig("cluster_latent_space.png")


def calculate_clustering_metrics(Y: np.ndarray, pre: np.ndarray) -> Tuple[float, float]:
    """Calculates clustering metrics (NMI and ARI)."""
    nmi = normalized_mutual_info_score(Y, pre)
    ari = adjusted_rand_score(Y, pre)
    return nmi, ari
