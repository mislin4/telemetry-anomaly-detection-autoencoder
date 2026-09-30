import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Simüle Edilmiş Çok Boyutlu Telemetri/Sinyal Verisi Üretimi
def generate_synthetic_telemetry(n_samples=5000, seed=42):
    np.random.seed(seed)
    time = np.linspace(0, 100, n_samples)
    
    ch1 = np.sin(0.1 * time) + np.random.normal(0, 0.05, n_samples)
    ch2 = np.cos(0.05 * time) * 2.0 + np.random.normal(0, 0.08, n_samples)
    ch3 = 0.02 * time + np.random.normal(0, 0.04, n_samples)
    ch4 = np.sin(0.2 * time) + np.cos(0.1 * time) + np.random.normal(0, 0.05, n_samples)
    
    data = np.stack([ch1, ch2, ch3, ch4], axis=1)
    
    anom_indices = np.random.choice(range(int(n_samples * 0.8), n_samples), size=50, replace=False)
    data[anom_indices] += np.random.uniform(2.5, 5.0, size=(50, 4))
    
    labels = np.zeros(n_samples)
    labels[anom_indices] = 1
    
    return data, labels

# 2. PyTorch Autoencoder Mimarisi
class TelemetryAutoencoder(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2):
        super(TelemetryAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 3. Eğitim ve Eşik Değeri Tespiti
def train_and_detect():
    X, y = generate_synthetic_telemetry()
    
    normal_mask = (y == 0)
    X_train_normal, X_test = train_test_split(X[normal_mask], test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_all_scaled = scaler.transform(X)
    
    tensor_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(tensor_train), batch_size=64, shuffle=True)
    
    model = TelemetryAutoencoder(input_dim=4, latent_dim=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(20):
        for batch in train_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32)
        reconstructed = model(all_tensor)
        reconstruction_error = torch.mean((all_tensor - reconstructed) ** 2, dim=1).numpy()
    
    threshold = np.mean(reconstruction_error[normal_mask]) + 3 * np.std(reconstruction_error[normal_mask])
    predictions = (reconstruction_error > threshold).astype(int)
    
    precision = np.sum((predictions == 1) & (y == 1)) / max(np.sum(predictions == 1), 1)
    recall = np.sum((predictions == 1) & (y == 1)) / max(np.sum(y == 1), 1)
    
    print(f"Eşik Değeri: {threshold:.4f}")
    print(f"Tespit Başarısı (Precision): {precision:.2f}")
    print(f"Duyarlılık (Recall): {recall:.2f}")

if __name__ == "__main__":
    train_and_detect()
