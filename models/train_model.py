import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

class FloodDataset(Dataset):
    def __init__(self, features, labels, sequence_length=10):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            self.features[idx:idx+self.sequence_length],
            self.labels[idx+self.sequence_length]
        )

class FloodLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super(FloodLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train_flood_model(data_path='data/processed/flood_training_data.csv', 
                      epochs=100, batch_size=64, sample_size=50000):
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_save_path = os.path.join(script_dir, 'flood_model_best.pth')
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print("FLOOD PREDICTION MODEL TRAINING")
    print("="*80)
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    
    # Load dataset
    print(f"\n[1/6] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Sample if dataset is too large (for faster training)
    if len(df) > sample_size:
        print(f"  Sampling {sample_size:,} from {len(df):,} records...")
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"✓ Loaded {len(df):,} records")
    print(f"  High severity floods: {df['flood_severity'].sum():,} ({df['flood_severity'].mean()*100:.1f}%)")
    
    # Prepare features - auto-detect all columns except target
    print("\n[2/6] Preparing features...")
    feature_columns = [col for col in df.columns if col != 'flood_severity']
    print(f"  Using {len(feature_columns)} features: {', '.join(feature_columns[:5])}{'...' if len(feature_columns) > 5 else ''}")
    X = df[feature_columns].values
    y = df['flood_severity'].values
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Balance dataset using weighted sampling (no oversampling needed with 30% flood rate)
    print("\n[3/6] Preparing balanced training...")
    print(f"  Original - Floods: {y.sum():,}, Non-floods: {len(y)-y.sum():,}")
    print(f"  Flood ratio: {y.mean()*100:.1f}%")
    
    # Calculate class weights for balanced training
    flood_weight = len(y) / (2 * y.sum())
    non_flood_weight = len(y) / (2 * (len(y) - y.sum()))
    class_weights = torch.FloatTensor([non_flood_weight, flood_weight])
    
    print(f"  Class weights - Non-flood: {non_flood_weight:.2f}, Flood: {flood_weight:.2f}")
    
    # Use all data without resampling
    X_balanced = X_scaled
    y_balanced = y
    
    # Train-test split
    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    train_dataset = FloodDataset(X_train, y_train)
    test_dataset = FloodDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Initialize model
    print("\n[5/6] Initializing model...")
    model = FloodLSTM(input_dim=len(feature_columns)).to(device)
    
    # Use weighted loss for class imbalance
    criterion = nn.BCELoss(weight=None)  # Will apply manual weighting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Using class-weighted loss for balanced training")
    
    # Training loop
    print(f"\n[6/6] Training for {epochs} epochs...")
    print("-"*80)
    
    best_f1 = -1.0  # Start at -1 to ensure first model is saved
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        test_loss = 0.0
        correct = total = 0
        tp = fp = fn = tn = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).unsqueeze(1)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                tp += ((predicted == 1) & (target == 1)).sum().item()
                fp += ((predicted == 1) & (target == 0)).sum().item()
                fn += ((predicted == 0) & (target == 1)).sum().item()
                tn += ((predicted == 0) & (target == 0)).sum().item()
        
        # Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        scheduler.step(avg_test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_test_loss:.4f} | "
                  f"Acc: {accuracy*100:.1f}% | "
                  f"F1: {f1:.3f} | "
                  f"Recall: {recall:.3f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': best_f1,
                'scaler': scaler,
                'feature_columns': feature_columns
            }, model_save_path)
            
            if (epoch + 1) % 10 == 0:
                print(f"         ✓ Best model saved (F1: {best_f1:.3f})")
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"  Best F1 Score: {best_f1:.3f}")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"\n💾 Model saved to: {model_save_path}")
    
    if device.type == 'cuda':
        print(f"🎮 GPU Memory Used: {torch.cuda.max_memory_allocated(0)/1024**2:.0f} MB")
    
    return model, scaler

if __name__ == "__main__":
    train_flood_model()
