import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ImprovedANFIS(nn.Module):
    """
    Gelişmiş Adaptif Ağ Tabanlı Bulanık Çıkarım Sistemi (ANFIS) modeli.

    Bu implementasyon, Gaussian üyelik fonksiyonları, çarpım tabanlı kural aktivasyonu,
    normalize edilmiş ateşleme güçleri ve Takagi-Sugeno tipi sonuç katmanı kullanır.
    """
    def __init__(self, n_inputs, n_rules):
        """
        Modelin başlatılması.

        Args:
            n_inputs (int): Giriş değişkenlerinin sayısı.
            n_rules (int): Bulanık kuralların sayısı.
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules

        # Gaussian üyelik fonksiyonlarının öğrenilebilir parametreleri
        self.mu = nn.Parameter(torch.randn(n_inputs, n_rules) * 0.5)  # Merkezler (başlangıç değerleri küçük)
        self.sigma = nn.Parameter(torch.exp(torch.randn(n_inputs, n_rules)))  # Genişlikler (pozitiflik garantili)

        # Takagi-Sugeno tipi sonuç katmanı (doğrusal fonksiyonlar)
        self.consequent = nn.Linear(n_inputs, n_rules, bias=True)
        nn.init.xavier_normal_(self.consequent.weight)  # Xavier başlatma ile dengeli gradyan akışı

    def forward(self, x):
        """
        İleri yayılım adımı.

        Args:
            x (torch.Tensor): Giriş tensörü (batch_size, n_inputs).

        Returns:
            torch.Tensor: Modelin çıktı tensörü (batch_size, 1).
        """
        # 1. Üyelik Derecesi Hesaplama (Gaussian)
        mu = self.mu.unsqueeze(0)  # (1, n_inputs, n_rules)
        sigma = torch.clamp(self.sigma.unsqueeze(0), min=1e-8)  # Numerik stabilite
        x_exp = x.unsqueeze(-1)  # (batch_size, n_inputs, 1)
        membership = torch.exp(-0.5 * ((x_exp - mu) / sigma)**2)

        # 2. Kural Aktivasyon Gücü (Çarpım Yöntemi)
        firing = membership.prod(dim=1)  # (batch_size, n_rules)
        firing = torch.clamp(firing, min=1e-8)  # Numerik stabilite

        # 3. Normalizasyon (Softmax Benzeri)
        norm_firing = firing / (firing.sum(dim=1, keepdim=True) + 1e-8)  # (batch_size, n_rules)

        # 4. Takagi-Sugeno Sonuç Katmanı (Doğrusal Kombinasyon ve Ağırlıklı Toplam)
        cq = self.consequent(x)  # (batch_size, n_rules) -> Her kural için lineer çıktı
        output = (norm_firing * cq).sum(dim=1, keepdim=True)  # Ağırlıklı toplam
        return output

def generate_data(n_samples, n_inputs=3):
    """
    Lineer bir ilişkiyi temsil eden sentetik veri üretir.

    Args:
        n_samples (int): Üretilecek örnek sayısı.
        n_inputs (int): Giriş değişkenlerinin sayısı.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Giriş (X) ve hedef (y) tensörleri.
    """
    X = np.random.randn(n_samples, n_inputs)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)  # Veri normalizasyonu
    y = np.sum(X * [0.8, 1.2, 1.0], axis=1, keepdims=True)  # Örnek lineer ilişki
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train(model, X_train, y_train, X_val, y_val, epochs=1000, patience=20):
    """
    ANFIS modelini eğitir.

    Args:
        model (nn.Module): Eğitilecek ANFIS modeli.
        X_train (torch.Tensor): Eğitim giriş verisi.
        y_train (torch.Tensor): Eğitim hedef verisi.
        X_val (torch.Tensor): Validasyon giriş verisi.
        y_val (torch.Tensor): Validasyon hedef verisi.
        epochs (int): Maksimum eğitim epok sayısı.
        patience (int): Erken durdurma için sabır epok sayısı.

    Returns:
        nn.Module: Eğitilmiş ANFIS modeli.
    """
    optimizer = torch.optim.Adam([
        {'params': model.mu, 'lr': 0.001},      # Merkezler için adaptif öğrenme oranı
        {'params': model.sigma, 'lr': 0.0005},   # Genişlikler için adaptif öğrenme oranı
        {'params': model.consequent.parameters(), 'lr': 0.01} # Sonuç katmanı için adaptif öğrenme oranı
    ], weight_decay=1e-4)  # Ağırlık azaltma (L2 regularizasyonu)
    criterion = nn.MSELoss()  # Ortalama Kare Hata kaybı
    best_loss = float('inf')
    train_losses, val_losses = [], []
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        reg_term = 0.01 * torch.norm(model.mu) + 0.005 * torch.norm(model.sigma) # Regularizasyon terimi
        total_loss = loss + reg_term

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradyan kırpma (stabilite için)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth') # En iyi modeli kaydet
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Erken durdurma: Epok {epoch}")
                break

        if epoch % 100 == 0:
            print(f"Epok {epoch} | Eğitim Kaybı: {loss.item():.4f} | Validasyon Kaybı: {val_loss.item():.4f}")

    # Kayıp grafiğini çizdir
    plt.plot(train_losses, label='Eğitim Kaybı')
    plt.plot(val_losses, label='Validasyon Kaybı')
    plt.xlabel('Epok')
    plt.ylabel('MSE Kaybı')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

def evaluate(model, X_test, y_test):
    """
    Eğitilmiş ANFIS modelini test verisi üzerinde değerlendirir.

    Args:
        model (nn.Module): Eğitilmiş ANFIS modeli.
        X_test (torch.Tensor): Test giriş verisi.
        y_test (torch.Tensor): Test hedef verisi.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = nn.MSELoss()(y_pred, y_test)
        r2 = 1 - mse / torch.var(y_test) # Belirleme katsayısı (R²)
        print(f"Test MSE: {mse.item():.4f}, R²: {r2.item():.4f}")

        # Tahmin-gerçek değer grafiğini çizdir
        plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.5)
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahminler')
        plt.title('Model Tahminleri vs. Gerçek Değerler')
        plt.plot([-3, 3], [-3, 3], 'r--') # Mükemmel tahmin çizgisi
        plt.grid(True)
        plt.show()

def plot_membership(model, input_idx=0):
    """
    Belirli bir giriş değişkeni için öğrenilmiş Gaussian üyelik fonksiyonlarını görselleştirir.

    Args:
        model (nn.Module): Eğitilmiş ANFIS modeli.
        input_idx (int): Görselleştirilecek giriş değişkeninin indeksi (0-based).
    """
    x = torch.linspace(-3, 3, 100)
    with torch.no_grad():
        mu = model.mu[input_idx].cpu()
        sigma = torch.clamp(model.sigma[input_idx], min=1e-8).cpu()
        plt.figure(figsize=(10, 5))
        for r in range(model.n_rules):
            plt.plot(x.numpy(), torch.exp(-0.5 * ((x - mu[r]) / sigma[r])**2).numpy(), label=f'Kural {r+1}')
        plt.title(f'Giriş {input_idx+1} Üyelik Fonksiyonları')
        plt.xlabel('Giriş Değeri')
        plt.ylabel('Üyelik Derecesi')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Hiperparametreler
    n_inputs = 3
    n_rules = 5
    n_samples_train = 1000
    n_samples_val = 200
    n_samples_test = 200
    n_epochs = 1000
    early_stopping_patience = 20

    # Veri üretimi
    X_train, y_train = generate_data(n_samples_train, n_inputs)
    X_val, y_val = generate_data(n_samples_val, n_inputs)
    X_test, y_test = generate_data(n_samples_test, n_inputs)

    # Model oluşturma ve eğitim
    model = ImprovedANFIS(n_inputs, n_rules)
    trained_model = train(model, X_train, y_train, X_val, y_val, epochs=n_epochs, patience=early_stopping_patience)

    # Modelin değerlendirilmesi
    evaluate(trained_model, X_test, y_test)

    # İlk giriş değişkeni için üyelik fonksiyonlarını görselleştirme
    plot_membership(trained_model, input_idx=0)
