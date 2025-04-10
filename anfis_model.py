import torch
import torch.nn as nn

class ANFIS(nn.Module):
    def __init__(self, n_inputs, n_rules):
        super().__init__()
        # Membership function parameters
        self.mu = nn.Parameter(torch.rand(n_inputs, n_rules)*2 -1)  # [-1,1] range
        self.sigma = nn.Parameter(torch.ones(n_inputs, n_rules))    # Positive
        
        # Rule layers
        self.rule_weights = nn.Linear(n_rules, n_rules, bias=False)
        self.defuzzify = nn.Linear(n_rules, 1)
        
    def forward(self, x):
        # Calculate Gaussian membership
        x = x.unsqueeze(-1)  # (batch, n_inputs) -> (batch, n_inputs, 1)
        mu = self.mu.unsqueeze(0)  # (1, n_inputs, n_rules)
        sigma = self.sigma.unsqueeze(0)
        membership = torch.exp(-0.5*((x-mu)/sigma)**2)  # (batch, n_inputs, n_rules)
        
        # Rule activation
        firing = membership.prod(dim=1)  # (batch, n_rules)
        normalized = torch.softmax(self.rule_weights(firing), dim=1)
        
        # Defuzzification
        return self.defuzzify(normalized * firing)

# Training loop example
def train(model, X, y, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        y_pred = model(X)
        loss = nn.MSELoss()(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}")
