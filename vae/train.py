import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

import clustering as _c
import generate_data as _d
import model as _m
import numpy as np
import generate as _g
import seeding as _s

proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_path)

from config.config import config as c


_s.set_seed(20)


def train(
    epoch: int,
    train_loader: DataLoader,
    model: _m.MyModel,
    optimizer: optim.Optimizer,
    val_loader: DataLoader,
    device: torch.device,
    GAMMA: float,
) -> float:
    model.train()
    train_loss = 0.0  # Initialize as float
    total_re_loss = 0.0  # Initialize as float
    total_kld_loss = 0.0  # Initialize as float

    n_iter = len(train_loader)
    L = _c.generate_gamma_schedule(n_iter, GAMMA)

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        beta = L[batch_idx]
        re_loss, kld_loss, loss = model.loss_function(
            recon_batch, data, mu, logvar, beta
        )

        if torch.isnan(loss).any():
            print("NaN Loss detected!")
            print(f"RE Loss: {re_loss.item()}, KL-D Loss: {kld_loss.item()}")
            break

        loss.backward()
        # add
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * data.size(0)  # Accumulate raw loss
        total_re_loss += re_loss.item() * data.size(0)  # Accumulate raw re_loss
        total_kld_loss += kld_loss.item() * data.size(0)  # Accumulate raw kld_loss

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * data.size(0)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    avg_train_loss = train_loss / len(train_loader.dataset)  # Scale only once
    avg_re_loss = total_re_loss / len(train_loader.dataset)  # Scale only once
    avg_kld_loss = total_kld_loss / len(train_loader.dataset)  # Scale only once

    print(
        f"====> Epoch: {epoch} Average Loss: {avg_train_loss:.4f} "
        f"(RE: {avg_re_loss:.4f}, KLD: {avg_kld_loss:.4f})"
    )

    validate(model, val_loader, device, GAMMA)

    return avg_train_loss  # Return the float directly


def validate(
    model: _m.MyModel,
    val_loader: DataLoader,
    device: torch.device,
    GAMMA: float,
) -> float:
    model.eval()
    total_val_loss = 0.0  # Initialize as float
    all_latents = []
    all_labels = []
    all_clusters = []

    n_iter = len(val_loader)
    L = _c.generate_gamma_schedule(n_iter, GAMMA)
    with torch.no_grad():
        for i, (data, labels) in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            beta = L[i]
            _, _, loss = model.loss_function(recon_batch, data, mu, logvar, beta)
            total_val_loss += loss.item() * data.size(0)  # Accumulate raw loss

            z_mu, _ = model.encode(data)
            all_latents.append(z_mu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_clusters.append(model.predict(data))

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_clusters = np.concatenate(all_clusters, axis=0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)  # Scale only once

    # If we have ground truth labels, calculate supervised metrics
    if len(np.unique(all_labels)) > 1:  # Ensure we have meaningful labels
        ari = adjusted_rand_score(all_labels, all_clusters)
        nmi = normalized_mutual_info_score(all_labels, all_clusters)
        print(f"====> Adjusted Rand Index: {ari:.4f}")
        print(f"====> Normalized Mutual Information: {nmi:.4f}")

    # Calculate unsupervised metrics (internal clustering metrics)
    if len(all_latents) > max(
        model.nClusters, 2
    ):  # Need enough samples for valid metrics
        try:
            silhouette = silhouette_score(all_latents, all_clusters)
            db_score = davies_bouldin_score(all_latents, all_clusters)
            ch_score = calinski_harabasz_score(all_latents, all_clusters)
            print(f"====> Silhouette Score: {silhouette:.4f}")
            print(f"====> Davies-Bouldin Score: {db_score:.4f} (lower is better)")
            print(f"====> Calinski-Harabasz Score: {ch_score:.1f} (higher is better)")
        except Exception as e:
            print(f"Warning: Could not calculate some clustering metrics. Error: {e}")

    # Calculate cluster distribution
    unique_clusters, cluster_counts = np.unique(all_clusters, return_counts=True)
    print("====> Cluster distribution:")
    for cluster, count in zip(unique_clusters, cluster_counts):
        print(
            f"      Cluster {cluster}: {count} samples ({100*count/len(all_clusters):.2f}%)"
        )

    print(f"====> Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss  # Return the float directly
    print(f"====> Validation Loss: {avg_val_loss:.4f}")
    # ... (rest of validate function - clustering metrics, etc.)
    return avg_val_loss  # Return the float directly


def test(
    model: _m.MyModel, test_loader: DataLoader, GAMMA: float, device: torch.device
) -> Tuple[float, Dict[str, List[float]]]:
    model.eval()
    test_loss = 0.0  # Initialize as float
    loss_histories = {
        "kld": [],
        "mse": [],
        "reconst": [],
        "loss_function": [],
    }
    n_iter = len(test_loader)
    L = _c.generate_gamma_schedule(n_iter, GAMMA)
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)

            recon_batch, mu, logvar = model(data)
            reconst, kld_loss, loss = model.loss_function(
                recon_batch, data, mu, logvar, L[i]
            )
            test_loss += loss.item() * data.size(0)  # Accumulate raw loss
            losses = _g.compute_losses(reconst, kld_loss, loss)  # Use gamma
            for key, value in losses.items():
                loss_histories[key].append(value.item())

            if i == 0:
                n = min(data.size(0), 16)
                _, C, H, W = data.shape  # Get image dimensions dynamically
                _recon_batch = recon_batch.view(-1, C, H, W)[
                    :n
                ]  # Reshape with correct dims
                comparison = torch.cat([data[:n], _recon_batch])

    test_loss /= len(test_loader.dataset)  # Scale only once
    print(f"====> Test set loss: {test_loss:.4f}")
    return test_loss, loss_histories


def save_best_model(
    epoch: int, model: _m.MyModel, best_loss: float, save_path: str
) -> None:
    """
    Saves the model state if a better loss is achieved.
    """
    print(f"Saving model at epoch {epoch} with Validation Loss: {best_loss:.4f}")
    print("=" * 50)  # Visual separator for clarity in output
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}\n")


def log_epoch_results(epoch: int, train_loss: float, val_loss: float) -> None:
    """
    Logs the training and validation loss for the current epoch.
    """
    print(f"Epoch {epoch}:")
    print(f"    Training Loss     : {train_loss:.4f}")
    print(f"    Validation Loss   : {val_loss:.4f}")


def train_and_evaluate(
    epochs: int,
    model: _m.MyModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: optim.Optimizer,
    save_path: str,
    patience: int,
    device: torch.device,
    GAMMA: float,
) -> Tuple[List[float], List[float]]:
    """
    Trains and evaluates the model over a specified number of epochs.
    Tracks the best validation loss and saves the model accordingly.
    Implements early stopping if the validation loss doesn't improve for a specified number of epochs.
    """
    best_loss = None
    best_epoch = 0
    train_loss_arr = []
    test_loss_arr = []
    no_improvement_counter = 0  # Tracks epochs without improvement
    loss_history_mean = {
        "mse": [],
        "reconst": [],
        "kld": [],
        "loss_function": [],
    }
    best_loss_history = {}
    model.freeze_prior()
    for epoch in range(1, epochs + 1):
        # Unfreeze prior after 10 epochs
        if epoch == 11:
            model.unfreeze_prior()
        # Train the model and record the training loss
        train_loss = train(
            epoch,
            train_loader,
            model,
            optimizer,
            validation_loader,
            device,
            GAMMA=GAMMA,
        )
        train_loss_arr.append(train_loss)

        # Test the model and record the validation loss
        test_loss, loss_history = test(
            model=model, test_loader=test_loader, GAMMA=GAMMA, device=device
        )  # Assume test_fn() returns a float
        test_loss_arr.append(test_loss)
        for key in loss_history:
            if loss_history[key]:  # Ensure the list is not empty
                mean_value = np.mean(loss_history[key])
                loss_history_mean[key].append(mean_value)

        # Log results for this epoch
        log_epoch_results(epoch, train_loss, test_loss)

        # Check if this is the best model so far
        if best_loss is None or test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            save_best_model(best_epoch, model, best_loss, save_path)
            best_loss_history = loss_history
            no_improvement_counter = 0  # Reset counter on improvement
        else:
            no_improvement_counter += 1

        # Check for NaN in validation loss
        if np.isnan(test_loss):
            print("Validation loss is NaN. Stopping training.")
            break

        # Early stopping if no improvement for `patience` epochs
        if no_improvement_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    # Log the best model results
    if best_loss is not None:
        print(
            f"Best model saved at epoch {best_epoch} with Validation Loss: {best_loss:.4f}."
        )
        print(f"Saved path: {save_path}")
    else:
        print("No model was saved.")

    _g.save_loss_plots(loss_history_mean, "loss_history_mean_per_epoch")
    _g.save_loss_plots(best_loss_history, "best_loss_history")
    return train_loss_arr, test_loss_arr


# Configuration (moved to the top)
GAMMA = 1e-5
GAMMA_STEP = 200
LATENT_DIM = 20
N_CLUSTERS = 13
N_CHANNELS = 1
BATCH_SIZE = c.get_batch_size()
SAVE_PATH = "./s3vdc_test4_3.pth"
PATIENCE = 10
# EPOCHS = 5  # Set your desired number of epochs
EPOCHS = 5000000  # Set your desired number of epochs

# Data loading
# N_CHANNELS = 3
# train_loader, val_loader, test_loader, trnSet = _d.get_house_data()
N_CHANNELS = 1
train_loader, val_loader, test_loader, trnSet = _d.get_anomaly_data()

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = _m.MyModel(
    latent_dim=LATENT_DIM, nClusters=N_CLUSTERS, nc=N_CHANNELS, ngf=128
).to(device)

# Initial gamma training
optimizer = optim.Adam(model.parameters(), lr=1e-3)
_c.gamma_training(model, train_loader, optimizer, GAMMA_STEP, GAMMA)

# Clustering initialization
_c.visualize_and_evaluate_clustering(
    model, train_loader, BATCH_SIZE, N_CLUSTERS, trnSet
)

# Training and evaluation loop
train_loss_arr, val_loss_arr = train_and_evaluate(
    epochs=EPOCHS,
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    validation_loader=val_loader,
    optimizer=optimizer,
    device=device,
    save_path=SAVE_PATH,
    patience=PATIENCE,
    GAMMA=GAMMA,
)

# Plotting loss curves
_g.plot_epoch_results(
    train_loss_arr, val_loss_arr, "training_validation_loss.png"
)  # Save plot with a name

# Final testing and analysis
model.load_state_dict(torch.load(SAVE_PATH))  # Load the best model
_g.test_analysis(test_loader, model, device)
_g.analyze_clusters(test_loader, model, N_CLUSTERS, device)
