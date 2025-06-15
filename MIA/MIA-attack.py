import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import numpy as np

# ------------------- Model Definitions -------------------
class DeeperCNN(nn.Module):  # More complex model to encourage overfitting
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class StrongerAttackModel(nn.Module):  # Bigger MLP
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ------------------- Utilities -------------------
def train_model(model, loader, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            pred = model(x).argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return correct / total

def get_probs(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.cuda()
            probs = F.softmax(model(x), dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)

def extract_features(probs):
    top1 = np.max(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    top2 = np.partition(probs, -2, axis=1)[:, -2]
    return np.stack([top1, entropy, top1 - top2], axis=1)

def prepare_attack_data(model, member_data, nonmember_data):
    member_loader = DataLoader(member_data, batch_size=64)
    nonmember_loader = DataLoader(nonmember_data, batch_size=64)
    member_probs = get_probs(model, member_loader)
    nonmember_probs = get_probs(model, nonmember_loader)
    X = np.vstack([extract_features(member_probs), extract_features(nonmember_probs)])
    y = np.array([1]*len(member_probs) + [0]*len(nonmember_probs))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ------------------- Main Execution -------------------
transform = transforms.ToTensor()
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
testset = datasets.MNIST('./data', train=False, transform=transform)
train_idx, shadow_idx = train_test_split(range(len(dataset)), test_size=0.5, random_state=42)

# Dataset splits
target_train = Subset(dataset, train_idx[:3000])
target_nonmem = Subset(dataset, train_idx[3000:6000])
shadow_train = Subset(dataset, shadow_idx[:3000])
shadow_nonmem = Subset(dataset, shadow_idx[3000:6000])

# Train shadow model
shadow_model = DeeperCNN().cuda()
train_model(shadow_model, DataLoader(shadow_train, batch_size=64, shuffle=True), epochs=10)

# Prepare shadow data
X_shadow, y_shadow = prepare_attack_data(shadow_model, shadow_train, shadow_nonmem)

# Train attack model
attack_model = StrongerAttackModel().cuda()
attack_loader = DataLoader(torch.utils.data.TensorDataset(X_shadow.cuda(), y_shadow.cuda()), batch_size=64, shuffle=True)
train_model(attack_model, attack_loader, epochs=15)

# Train target model
target_model = DeeperCNN().cuda()
train_model(target_model, DataLoader(target_train, batch_size=64, shuffle=True), epochs=10)
train_acc = evaluate(target_model, DataLoader(target_train, batch_size=64))
test_acc = evaluate(target_model, DataLoader(testset, batch_size=64))

# Attack evaluation
X_target, y_target = prepare_attack_data(target_model, target_train, target_nonmem)
with torch.no_grad():
    preds = attack_model(X_target.cuda()).argmax(1).cpu()
attack_precision = precision_score(y_target, preds, pos_label=1)

# Final Output
print("\n=== MIA Summary ===")
print(f"Target Model Training Accuracy:  {train_acc * 100:.2f}%")
print(f"Target Model Testing Accuracy:   {test_acc * 100:.2f}%")
print(f"Attack Model Precision (member): {attack_precision * 100:.2f}%")
