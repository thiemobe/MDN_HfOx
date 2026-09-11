import random
import numpy as np
import torch
import pandas as pd
import argparse
import json
import os
import joblib

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from MDN import MDN
from MDN_loss import mdn_loss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

DATA_FILE = 'data/train_val_samples_removed_outliers.csv'
N_SAMPLES = 500                 # training samples per model
N_RUNS = 5                      # number of models trained with different shuffles
OUTPUT_DIR = f'models/n{N_SAMPLES}'               # None -> auto: runs/multi_shuffle_logarithmic_<n_samples>samples
VAL_RATIO = 0.2                 # validation size as fraction of n_samples
EARLY_STOPPING_PATIENCE = 200   # epochs without improvement before stopping
EARLY_STOPPING_MIN_DELTA = 1e-4
WEIGHT_INIT_SEED = 42            # FIXED weight initialization seed (same for all models)

MODEL_CONFIG = {
    'hidden_layers': 1,
    'hidden_dim': 64,
    'n_components': 2,
    'optimizer': 'Adam',
    'dropout': 0.3,
    'lr': 3e-3,
    'epochs': 800,
    'loss_type': 'crps'  # 'crps'
}

# Deterministic shuffle seeds per run (same logic for any N)
# Format: shuffle_seed = n_samples * 1000 + run_index
# Examples:
#   50 samples, run 0 → seed 50000
#   50 samples, run 1 → seed 50001
#   500 samples, run 0 → seed 500000
#   500 samples, run 1 → seed 500001


def train_mdn_with_early_stopping(model, train_loader, val_loader,
                                   epochs=MODEL_CONFIG['epochs'], lr=MODEL_CONFIG['lr'],
                                   optimizer_name=MODEL_CONFIG['optimizer'],
                                   verbose=True, writer=None, loss_type=MODEL_CONFIG['loss_type'],
                                   early_stopping_patience=EARLY_STOPPING_PATIENCE,
                                   early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA):
    """
    Wrapper around train_mdn that adds early stopping based on validation loss.
    
    Parameters
    ----------
    early_stopping_patience : int
        Number of epochs without improvement before stopping
    early_stopping_min_delta : float
        Minimum change in validation loss to qualify as improvement
    
    Returns
    -------
    model, history
        Trained model and training history
    """
    
    # Train with early stopping by monitoring validation loss
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    from torch.optim import Adam, SGD
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'early_stopped': False,
        'stopped_epoch': 0,
        'best_epoch': 0,
        'patience': early_stopping_patience
    }
    
    print(f"  Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta:.1e}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pi, mu, sigma = model(batch_X)
            
            # Compute loss (use same loss_type as original)
            loss = mdn_loss(pi, mu, sigma, batch_y, loss_type=loss_type)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * batch_X.shape[0]
            train_count += batch_X.shape[0]
        
        train_loss = train_loss_sum / train_count
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                output = model(batch_X)
                pi, mu, sigma = output
                
                loss = mdn_loss(pi, mu, sigma, batch_y, loss_type=loss_type)
                
                val_loss_sum += loss.item() * batch_X.shape[0]
                val_count += batch_X.shape[0]
        
        val_loss = val_loss_sum / val_count
        history['val_loss'].append(val_loss)
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Early stopping check
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            best_state = model.state_dict().copy()
            
            if verbose and (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} (best)")
        else:
            patience_counter += 1
            
            if verbose and (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}: validation loss did not improve for {early_stopping_patience} epochs")
                print(f"  Best model was at epoch {best_epoch+1} with val_loss={best_val_loss:.6f}")
                
                # Restore best model
                model.load_state_dict(best_state)
                
                history['early_stopped'] = True
                history['stopped_epoch'] = epoch + 1
                history['best_epoch'] = best_epoch + 1
                history['best_val_loss'] = best_val_loss
                
                if writer:
                    writer.add_scalar('EarlyStop/stopped_epoch', epoch + 1, 0)
                    writer.add_scalar('EarlyStop/best_epoch', best_epoch + 1, 0)
                
                break
    
    if not history['early_stopped']:
        history['best_epoch'] = best_epoch + 1
        history['best_val_loss'] = best_val_loss
        print(f"  Training completed all {epochs} epochs (no early stopping)")
    
    return model, history


def build_data(csv_file=DATA_FILE,
               n_total=N_SAMPLES):
    """
    Load and return first n_total rows from CSV.
    
    Parameters
    ----------
    n_total : int
        Total rows to load (= n_train_samples + n_val_samples)
    """
    print(f"Loading CSV file: '{csv_file}'")
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    total_rows = len(df)
    if n_total > total_rows:
        print(f"WARNING: Requested {n_total} samples but CSV only has {total_rows}. Using all {total_rows}.")
        n_total = total_rows
    
    df = df.iloc[:n_total]  # Take first n_total
    
    # Extract the data array
    data_array = df.to_numpy()
    print(f"Loaded data shape: {data_array.shape}")

    # Column handling like before
    state_str = df.iloc[:, 0].astype(str).str.strip().str.lower()
    state_bin = np.where(state_str == 'set', 1.0, 0.0)

    G_col9 = pd.to_numeric(df.iloc[:, 9], errors='coerce').to_numpy(dtype=float) * 1e6
    V_col4 = pd.to_numeric(df.iloc[:, 3], errors='coerce').to_numpy(dtype=float)
    conductance_after = pd.to_numeric(df.iloc[:, [11, 14, 17]].mean(axis=1), errors='coerce').to_numpy(dtype=float) * 1e6

    X = np.column_stack([state_bin, G_col9, V_col4]).astype(float)
    y = conductance_after.astype(float)
    return X, y


def train_multi_shuffle(params, 
                        csv_file=DATA_FILE,
                        n_samples=N_SAMPLES,
                        n_runs=N_RUNS,
                        output_dir=OUTPUT_DIR,
                        weight_init_seed=WEIGHT_INIT_SEED,
                        train_val_split_ratio=VAL_RATIO,
                        early_stopping_patience=EARLY_STOPPING_PATIENCE,
                        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA):
    """
    Train N models on the SAME n_samples but with DIFFERENT shuffles with early stopping.
    
    All models use the SAME weight initialization seed.
    Each model trains on a different random shuffle of the data.
    Each model is evaluated on a held-out validation set with early stopping.
    
    Waterproof shuffling:
        shuffle_seed_for_run_i = n_samples * 1000 + i
    
    This ensures:
    - Reproducibility (same n_samples, run i always gets same shuffle)
    - Different shuffles across runs (seed changes with run_index)
    - Scalability (same logic works for 50, 100, 150, ..., 500 samples)
    
    Parameters
    ----------
    train_val_split_ratio : float
        Validation ratio as fraction of training samples (default: 0.2 for 20% val)
        Example: n_samples=100 with ratio=0.2 → load 120 total (100 train + 20 val)
    early_stopping_patience : int
        Number of epochs without validation improvement before stopping (default: 20)
    early_stopping_min_delta : float
        Minimum change in validation loss to qualify as improvement (default: 1e-4)
    """

    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate validation size (as fraction of training samples, not split of total)
    n_train = n_samples  # Training samples (main parameter)
    n_val = max(1, int(n_samples * train_val_split_ratio))  # At least 1 val sample
    n_total = n_samples + n_val  # Total to load from CSV
    
    print("\n" + "=" * 80)
    print(f"TRAINING {n_runs} MODELS ON {n_samples} TRAINING SAMPLES")
    print(f"Per-Model Split: {n_samples} train + {n_val} val ({train_val_split_ratio*100:.0f}% val)")
    print(f"Total samples per model: {n_total}")
    print(f"Weight Initialization Seed (all models): {weight_init_seed}")
    print(f"Shuffle Seeds: {[n_samples * 1000 + i for i in range(n_runs)]}")
    print("=" * 80)
    
    # Load full data once (load n_total = n_samples + n_val)
    X, y = build_data(csv_file, n_total=n_total)
    
    print(f"\nData loaded: X.shape={X.shape}, y.shape={y.shape}")
    
    # Fit scalers once on full dataset (stable reference for all models)
    print("\nFitting scalers on full dataset...")
    scaler_X = StandardScaler()
    X_cont = X[:, 1:]  # continuous features: G_before, V_applied
    X_cont_scaled = scaler_X.fit_transform(X_cont)
    X_scaled = np.column_stack([X[:, 0], X_cont_scaled])
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    print(f"Scalers fitted (X_mean={scaler_X.mean_}, X_scale={scaler_X.scale_})")
    
    # Train each model
    for run_idx in range(n_runs):
        print(f"\n" + "-" * 80)
        print(f"Training Model {run_idx + 1}/{n_runs}")
        print("-" * 80)
        
        # Deterministic shuffle seed for this run (WATERPROOF FORMULA)
        shuffle_seed = n_samples * 1000 + run_idx
        print(f"Shuffle seed: {shuffle_seed}")
        
        # Set seed for data shuffling
        np.random.seed(shuffle_seed)
        shuffle_indices = np.random.permutation(len(X_scaled))
        X_shuffled = X_scaled[shuffle_indices]
        y_shuffled = y_scaled[shuffle_indices]
        
        # Split: first n_samples for training, next n_val for validation
        X_train = X_shuffled[:n_samples]
        y_train = y_shuffled[:n_samples]
        X_val = X_shuffled[n_samples:n_samples+n_val]
        y_val = y_shuffled[n_samples:n_samples+n_val]
        
        print(f"\n  Train set: {X_train.shape[0]} samples")
        print(f"  Val set:   {X_val.shape[0]} samples")
        
        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        
        # Set weight initialization seed (same for all models)
        torch.manual_seed(weight_init_seed)
        np.random.seed(weight_init_seed)
        random.seed(weight_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(weight_init_seed)
            torch.cuda.manual_seed_all(weight_init_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Create DataLoader (use shuffle_seed for DataLoader shuffling too, for consistency)
        g = torch.Generator()
        g.manual_seed(shuffle_seed)
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), 
            batch_size=32, shuffle=True, generator=g
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t), 
            batch_size=64, shuffle=False
        )
        
        # Create model
        model = MDN(
            input_dim=3,
            hidden_dim=params['hidden_dim'],
            n_components=params.get('n_components', 2),
            num_hidden_layers=params['hidden_layers'],
            dropout_rate=params.get('dropout', 0.3)
        )
        
        # TensorBoard
        tb_name = f"runs/tb_shuffle_{n_samples}samples_run{run_idx + 1}_seed{shuffle_seed}"
        writer = SummaryWriter(log_dir=tb_name)
        
        # Train with early stopping
        print(f"\n  Training with early stopping (patience={early_stopping_patience} epochs)")
        model, history = train_mdn_with_early_stopping(
            model, train_loader, val_loader,
            epochs=params['epochs'],
            lr=params['lr'],
            optimizer_name=params['optimizer'],
            verbose=True,
            writer=writer,
            loss_type=params.get('loss_type', 'crps'),
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta
        )
        
        writer.flush()
        writer.close()
        
        # Save model
        model_path = os.path.join(
            output_dir,
            f'model_run{run_idx + 1}_shuffle_seed{shuffle_seed}_{n_samples}samples_logarithmic.pt'
        )
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "input_dim": 3,
                "hidden_dim": params['hidden_dim'],
                "n_components": params.get('n_components', 2),
                "hidden_layers": params['hidden_layers'],
                "dropout": params.get('dropout', 0.3)
            },
            "n_samples": n_samples,
            "n_train": X_train.shape[0],
            "n_val": X_val.shape[0],
            "run_index": run_idx,
            "weight_init_seed": weight_init_seed,
            "shuffle_seed": shuffle_seed,
            "train_val_split_ratio": train_val_split_ratio,
            "shuffled_indices": shuffle_indices.tolist(),
            "scalers": {
                "X_mean": scaler_X.mean_.tolist(),
                "X_scale": scaler_X.scale_.tolist(),
                "y_mean": scaler_y.mean_.tolist(),
                "y_scale": scaler_y.scale_.tolist(),
            }
        }, model_path)
        
        print(f"  Model saved: {model_path}")
    
    # Save scalers
    scaler_X_path = os.path.join(output_dir, f'scaler_X_{n_samples}samples_logarithmic.joblib')
    scaler_y_path = os.path.join(output_dir, f'scaler_y_{n_samples}samples_logarithmic.joblib')
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"\n  Saved scalers to {output_dir}")
    
    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "n_train": n_train,
        "n_val": n_val,
        "train_val_split_ratio": train_val_split_ratio,
        "n_runs": n_runs,
        "weight_init_seed": weight_init_seed,
        "shuffle_seeds": [n_samples * 1000 + i for i in range(n_runs)],
        "shuffle_formula": "shuffle_seed = n_samples * 1000 + run_index",
        "config": params,
        "output_directory": output_dir,
        "created_at": datetime.now().isoformat()
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE: {n_runs} models trained on {n_samples} shuffled samples")
    print(f"  Train/Val Split: {n_train} train / {n_val} val samples per model")
    print(f"  TensorBoard logs: runs/tb_shuffle_*")
    print("=" * 80 + "\n")
    
    return scaler_X, scaler_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multiple MDN models on same samples with DIFFERENT shuffles"
    )
    parser.add_argument(
        '--data-file', 
        type=str, 
        default=DATA_FILE,
        help=f'Path to CSV data file (default: {DATA_FILE})'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=N_SAMPLES,
        help=f'Number of samples to train on (default: {N_SAMPLES})'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=N_RUNS,
        help=f'Number of models to train with different shuffles (default: {N_RUNS})'
    )
    parser.add_argument(
        '--weight-init-seed',
        type=int,
        default=WEIGHT_INIT_SEED,
        help=f'Weight initialization seed (same for all models, default: {WEIGHT_INIT_SEED})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help='Output directory'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=VAL_RATIO,
        help=f'Validation ratio as fraction of training samples (default: {VAL_RATIO} for {int(VAL_RATIO * 100)}%%)'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f'Early stopping patience: epochs without improvement before stopping (default: {EARLY_STOPPING_PATIENCE})'
    )
    parser.add_argument(
        '--early-stopping-min-delta',
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
        help=f'Early stopping min delta: minimum change in validation loss to qualify as improvement (default: {EARLY_STOPPING_MIN_DELTA})'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {args.n_samples}")
    if args.n_runs <= 0:
        raise ValueError(f"n_runs must be > 0, got {args.n_runs}")
    
    print("\n" + "=" * 80)
    print("MULTI-SHUFFLE MDN TRAINING")
    print("=" * 80)
    print(f"Data file: {args.data_file}")
    print(f"Training samples per model: {args.n_samples}")
    print(f"Validation ratio: {args.val_ratio*100:.0f}% of training samples")
    print(f"Number of models: {args.n_runs}")
    print(f"Weight init seed (all models): {args.weight_init_seed}")
    print(f"Shuffle formula: shuffle_seed = n_samples * 1000 + run_index")
    print("=" * 80 + "\n")
    
    # Train ensemble
    scaler_X, scaler_y = train_multi_shuffle(
        MODEL_CONFIG,
        csv_file=args.data_file,
        n_samples=args.n_samples,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        weight_init_seed=args.weight_init_seed,
        train_val_split_ratio=args.val_ratio,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta
    )
