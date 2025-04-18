import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ImprovedANFIS(nn.Module):
    def __init__(self, n_inputs, n_rules):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules

        # Daha iyi parametre başlatma
        self.mu = nn.Parameter(torch.randn(n_inputs, n_rules) * 0.5)
        self.sigma = nn.Parameter(torch.exp(torch.randn(n_inputs, n_rules)))  # Pozitif sigma garantisi

        # Takagi-Sugeno sonuç katmanı
        self.consequent = nn.Linear(n_inputs, n_rules, bias=True)
        nn.init.normal_(self.consequent.weight, mean=0.0, std=0.1)  # Ağırlık başlatma

    def forward(self, x):
        # Vektörize hesaplama
        mu = self.mu.unsqueeze(0)  # (1, n_inputs, n_rules)
        sigma = torch.clamp(self.sigma.unsqueeze(0), min=1e-4)
        x_exp = x.unsqueeze(-1)  # (batch, n_inputs, 1)
        
        # Gaussian üyelik fonksiyonları
        membership = torch.exp(-0.5 * ((x_exp - mu) / sigma)**2)
        firing = membership.prod(dim=1)  # (batch, n_rules)
        firing = torch.clamp(firing, min=1e-8)
        
        # Normalizasyon
        norm_firing = firing / (firing.sum(dim=1, keepdim=True) + 1e-8)
        
        # Sonuç katmanı
        cq = self.consequent(x)  # (batch, n_rules)
        return (norm_firing * cq).sum(dim=1, keepdim=True)

def generate_data(n_samples, n_inputs=3):
    X = np.random.randn(n_samples, n_inputs) * 1.5  # Normalize edilecek veri
    y = np.sum(X * [0.8, 1.2, 1.0], axis=1, keepdims=True)  # Lineer kombinasyon
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train(model, X_train, y_train, X_val, y_val, epochs=1000, patience=20):
    optimizer = torch.optim.Adam([
        {'params': model.mu, 'lr': 0.001},
        {'params': model.sigma, 'lr': 0.0005},
        {'params': model.consequent.parameters(), 'lr': 0.01}
    ], weight_decay=1e-4)
    
    best_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = nn.MSELoss()(y_pred, y_train)
        
        # Regularizasyon
        reg_term = 0.01 * torch.norm(model.mu) + 0.005 * torch.norm(model.sigma)
        total_loss = loss + reg_term
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradyan kırpma
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.MSELoss()(val_pred, y_val)
            
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    
    # Loss grafiği
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()
    
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = nn.MSELoss()(y_pred, y_test)
        r2 = 1 - mse / torch.var(y_test)
        print(f"Test MSE: {mse.item():.4f}, R²: {r2.item():.4f}")
        
        # Tahmin-Gerçek grafiği
        plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.5)
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahminler')
        plt.title('Model Performansı')
        plt.plot([-3,3], [-3,3], 'r--')
        plt.show()

# Veri hazırlama ve normalizasyon
n_inputs = 3
n_rules = 5
X_train, y_train = generate_data(1000, n_inputs)
X_val, y_val = generate_data(200, n_inputs)
X_test, y_test = generate_data(200, n_inputs)

# Model oluşturma ve eğitim
model = ImprovedANFIS(n_inputs, n_rules)
train(model, X_train, y_train, X_val, epochs=1000)

# Değerlendirme
evaluate(model, X_test, y_test)

# Üyelik fonksiyonlarını görselleştirme
def plot_membership(model, input_idx=0):
    x = torch.linspace(-3, 3, 100)
    with torch.no_grad():
        mu = model.mu[input_idx].cpu()
        sigma = torch.clamp(model.sigma[input_idx], min=1e-4).cpu()
        plt.figure(figsize=(10,5))
        for r in range(model.n_rules):
            plt.plot(x.numpy(), torch.exp(-0.5*((x - mu[r])/sigma[r])**2).numpy(), 
                     label=f'Rule {r+1}')
        plt.title(f'Input {input_idx+1} Üyelik Fonksiyonları')
        plt.xlabel('Giriş Değeri')
        plt.ylabel('Üyelik Derecesi')
        plt.legend()
        plt.show()

plot_membership(model, 0)
