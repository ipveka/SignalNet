"""
Training script for SignalNet.
"""
from signalnet.models.transformer import SignalTransformer
from signalnet.data.loader import SignalDataLoader
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
from typing import Optional
# Import the generator function directly
from signalnet.data.generate_data import generate_signal_data

def train(
    data_path: Optional[str] = None,
    context_length: int = 24,
    prediction_length: int = 6,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_model: bool = True
):
    print("========== SignalNet Training ==========")
    # Step 1: Data preparation
    # If no data_path is provided, generate synthetic data using the generator function
    if data_path is None:
        print("[INFO] No data_path provided. Generating synthetic data using the generator function...")
        generated_path = 'input/signal_data.csv'
        os.makedirs('input', exist_ok=True)
        generate_signal_data(n_series=20, length=200, output_file=generated_path)
        data_path = generated_path
        print(f"[INFO] Synthetic data generated at {data_path}")
    else:
        print(f"[INFO] Using data from: {data_path}")
    # Step 2: Load data and prepare train/test splits
    print("[INFO] Loading data and preparing train/test splits...")
    loader = SignalDataLoader(data_path, context_length, prediction_length)
    train_ds, test_ds = loader.train_test_split(test_size=0.2)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    print(f"[INFO] Number of training windows: {len(train_ds)}")
    print(f"[INFO] Number of test windows: {len(test_ds)}")
    # Print features used
    features_used = getattr(train_ds, 'features_used', None)
    if features_used is not None:
        print(f"[TRAIN] Features used in the model: {features_used}")
    else:
        print("[TRAIN] Features used in the model: ['day_of_week (normalized)', 'hour_of_day (normalized)', 'minute_of_hour (normalized)', 'month (normalized)', 'day_of_month (normalized)', 'is_weekend']")
    # Step 3: Model setup
    # Initialize the transformer model for signal prediction
    print("[INFO] Initializing SignalTransformer model...")
    model = SignalTransformer(
        input_dim=1,
        model_dim=32,
        num_heads=2,
        num_layers=2,
        output_dim=1,
        time_feat_dim=6
    )
    # Step 4: Device selection (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    model.to(device)
    # Step 5: Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # Step 6: Training loop
    print("[INFO] Starting training loop...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        total_loss = 0
        # Iterate over training batches
        for batch_idx, (context, target, context_time_feat, target_time_feat) in enumerate(train_dl):
            # Move data to device
            context = context.to(device)
            target = target.to(device)
            context_time_feat = context_time_feat.to(device)
            target_time_feat = target_time_feat.to(device)
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass (teacher forcing)
            output = model(context, context_time_feat, target_time_feat, target)
            # Compute loss
            loss = criterion(output, target)
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            total_loss += loss.item() * context.size(0)
            # Print batch loss every 10 batches or at the end
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dl):
                print(f"  [Batch {batch_idx+1}/{len(train_dl)}] Batch Loss: {loss.item():.4f}")
        # Compute and print average loss for the epoch
        avg_loss = total_loss / len(train_dl.dataset)  # type: ignore
        print(f"[INFO] Epoch {epoch+1} completed. Average Train Loss: {avg_loss:.4f}")
    # Step 7: Evaluation on test set
    print("\n[INFO] Training complete. Starting evaluation on test set...")
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch_idx, (context, target, context_time_feat, target_time_feat) in enumerate(test_dl):
            # Move data to device
            context = context.to(device)
            target = target.to(device)
            context_time_feat = context_time_feat.to(device)
            target_time_feat = target_time_feat.to(device)
            # Forward pass (no teacher forcing)
            output = model(context, context_time_feat, target_time_feat)
            # Compute loss
            loss = criterion(output, target)
            total_loss += loss.item() * context.size(0)
            # Print batch loss every 10 batches or at the end
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_dl):
                print(f"  [Test Batch {batch_idx+1}/{len(test_dl)}] Batch Loss: {loss.item():.4f}")
        # Compute and print average test loss
        avg_loss = total_loss / len(test_dl.dataset)  # type: ignore
        print(f"[INFO] Test Loss: {avg_loss:.4f}")
    # Step 8: Save the trained model weights (if enabled)
    if save_model:
        print("[INFO] Saving trained model weights to output/signalnet_model.pth...")
        os.makedirs('output', exist_ok=True)
        torch.save(model.state_dict(), 'output/signalnet_model.pth')
        print("[INFO] Model saved.")
    print("========== Training & Evaluation Finished ==========")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SignalNet on time series data.")
    parser.add_argument('--data_path', type=str, default=None, help='Path to input CSV data')
    parser.add_argument('--context_length', type=int, default=24)
    parser.add_argument('--prediction_length', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no_save_model', action='store_true', help='Do not save the trained model after training')
    args = parser.parse_args()
    train(
        args.data_path,
        args.context_length,
        args.prediction_length,
        args.epochs,
        args.batch_size,
        args.lr,
        save_model=not args.no_save_model
    )
