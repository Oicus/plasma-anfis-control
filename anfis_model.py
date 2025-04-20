import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ImprovedANFIS(nn.Module):
    """Geliştirilmiş Adaptif Ağ Tabanlı Bulanık Çıkarım Sistemi (ANFIS) modeli."""
    def __init__(self, n_inputs, n_rules):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules

        # Üyelik fonksiyonu parametreleri (Gaussian)
        self.mu = nn.Parameter(torch.randn(n_inputs, n_rules) * 0.5)  # Merkezler
        self.sigma = nn.Parameter(torch.exp(torch.randn(n_inputs, n_rules)))  # Genişlikler (pozitiflik garantili)

        # Takagi-Sugeno sonuç katmanı (doğrusal)
        self.consequent = nn.Linear(n_inputs, n_rules, bias=True)
        nn.init.normal_(self.consequent.weight, mean=0.0, std=0.1)  # Ağırlık başlatma

    def forward(self, x):
        """İleri yayılım adımı."""
        # Boyut ayarlamaları (batch, giriş, kural)
        mu = self.mu.unsqueeze(0)
        sigma = torch.clamp(self.sigma.unsqueeze(0), min=1e-8)  # Numerik stabilite
        x_exp = x.unsqueeze(-1)

        # Gaussian üyelik dereceleri
        membership = torch.exp(-0.5 * ((x_exp - mu) / sigma)**2)

        # Kural ateşleme güçleri (çarpım)
        firing = membership.prod(dim=1)
        firing = torch.clamp(firing, min=1e-8)  # Numerik stabilite

        # Normalizasyon
        norm_firing = firing / (firing.sum(dim=1, keepdim=True) + 1e-8)

        # Sonuç katmanı
        cq = self.consequent(x)

        # Ağırlıklı ortalama (bulanık çıkarım)
        return (norm_firing * cq).sum(dim=1, keepdim=True)

def generate_data(n_samples, n_inputs=3):
    """Sentetik veri üretir (lineer ilişki)."""
    X = np.random.randn(n_samples, n_inputs) * 1.5
    y = np.sum(X * [0.8, 1.2, 1.0], axis=1, keepdims=True)
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train(model, X_train, y_train, X_val, y_val, epochs=1000, patience=20):
    """Modeli eğitir."""
    optimizer = torch.optim.Adam([
        {'params': model.mu, 'lr': 0.001},
        {'params': model.sigma, 'lr': 0.0005},
        {'params': model.consequent.parameters(), 'lr': 0.01}
    ], weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    train_losses, val_losses = [], []
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        reg_term = 0.01 * torch.norm(model.mu) + 0.005 * torch.norm(model.sigma)
        total_loss = loss + reg_term

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Erken durdurma: Epok {epoch}")
                break

        if epoch % 100 == 0:
            print(f"Epok {epoch} | Train Kaybı: {loss.item():.4f} | Val Kaybı: {val_loss.item():.4f}")

    plt.plot(train_losses, label='Eğitim Kaybı')
    plt.plot(val_losses, label='Validasyon Kaybı')
    plt.xlabel('Epok')
    plt.ylabel('MSE Kaybı')
    plt.legend()
    plt.show()

    return model

def evaluate(model, X_test, y_test):
    """Modeli değerlendirir."""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = nn.MSELoss()(y_pred, y_test)
        r2 = 1 - mse / torch.var(y_test)
        print(f"Test MSE: {mse.item():.4f}, R²: {r2.item():.4f}")
        plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.5)
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahminler')
        plt.title('Model Performansı')
        plt.plot([-3, 3], [-3, 3], 'r--')
        plt.show()

def plot_membership(model, input_idx=0):
    """Bir giriş değişkeni için üyelik fonksiyonlarını görselleştirir."""
    x = torch.linspace(-3, 3, 100)
    with torch.no_grad():
        mu = model.mu[input_idx].cpu()
        sigma = torch.clamp(model.sigma[input_idx], min=1e-8).cpu()
        plt.figure(figsize=(10, 5))
        for r in range(model.n_rules):
            plt.plot(x.numpy(), torch.exp(-0.5 * ((x - mu[r]) / sigma[r])**2).numpy(),
                     label=f'Kural {r+1}')
        plt.title(f'Giriş {input_idx+1} Üyelik Fonksiyonları')
        plt.xlabel('Giriş Değeri')
        plt.ylabel('Üyelik Derecesi')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    n_inputs = 3
    n_rules = 5
    X_train, y_train = generate_data(1000, n_inputs)
    X_val, y_val = generate_data(200, n_inputs)
    X_test, y_test = generate_data(200, n_inputs)

    model = ImprovedANFIS(n_inputs, n_rules)
    trained_model = train(model, X_train, y_train, X_val, y_val, epochs=1000)
    evaluate(trained_model, X_test, y_test)
    plot_membership(trained_model, 0)
