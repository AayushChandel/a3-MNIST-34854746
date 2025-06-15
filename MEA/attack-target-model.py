import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import numpy as np
import random
import time
from sklearn.metrics import accuracy_score
import torch.nn as nn
#importing the target model as required by the assignment
from MIA.a3_mnist import Lenet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_model = Lenet().to(device)
target_model.load_state_dict(torch.load("defended_target_model.pt", map_location=device))
target_model.eval()

# creating a black box environment to generate soft labels
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform) #MNIST dataset

query_indices = random.sample(range(len(test_dataset)), 6000) # number of datasets
query_subset = Subset(test_dataset, query_indices)
query_loader = DataLoader(query_subset, batch_size=64, shuffle=False)

X_query = []
Y_query_soft = []
Y_true = []

with torch.no_grad():
    for data, labels in query_loader:
        data = data.to(device)
        outputs = target_model(data)
        probs = torch.exp(outputs).cpu().numpy()
        X_query.append(data.view(data.size(0), -1).cpu().numpy())
        Y_query_soft.append(probs)
        Y_true.extend(labels.cpu().numpy())

X_query = np.vstack(X_query).astype(np.float32)
Y_query_soft = np.vstack(Y_query_soft).astype(np.float32)
Y_true = np.array(Y_true)

#creating Attacker model to get images
class AttackerMLP(nn.Module):
    def __init__(self):
        super(AttackerMLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

attacker_model = AttackerMLP().to(device)

#Define the optimizer for model training
optimizer = torch.optim.Adam(attacker_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_tensor = torch.tensor(X_query).to(device)
Y_tensor = torch.tensor(Y_query_soft).to(device)

#training the model with soft labels recieved from target model
epochs = 30
batch_size = 64

for epoch in range(epochs):
    attacker_model.train()
    permutation = torch.randperm(X_tensor.size(0))
    for i in range(0, X_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_tensor[indices], Y_tensor[indices]
        optimizer.zero_grad()
        outputs = attacker_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        

#Evaluate Surrogate Model
eval_start_time = time.time()

attacker_model.eval()
with torch.no_grad():
    # Predicted probabilities and labels from attacker
    pred_soft = attacker_model(X_tensor).cpu().numpy()
    pred_labels = np.argmax(pred_soft, axis=1)
    
    # Labels from target model outputs (ground truth for attacker)
    target_labels = np.argmax(Y_query_soft, axis=1)

    # Metrics
    acc = accuracy_score(Y_true, pred_labels)                 # Accuracy vs true labels = 1 - R_test
    agreement = accuracy_score(target_labels, pred_labels)    # Agreement with target model = 1 - R_unif

    # Average MSE loss
    mse_loss_fn = nn.MSELoss()
    avg_loss = mse_loss_fn(torch.tensor(pred_soft), torch.tensor(Y_query_soft)).item()

eval_end_time = time.time()
elapsed_time = eval_end_time - eval_start_time

# === Output Results ===
print("\n--- Final Attack Evaluation ---")
print(f"1 − R_test        : {acc * 100:.2f}%")
print(f"1 − R_unif        : {agreement * 100:.2f}%")
print(f"Average MSE Loss  : {avg_loss:.6f}")
print(f"Final Batch Loss  : {loss.item():.6f}")
print(f"Queries Used      : {len(X_query)}")
print(f"Evaluation Time   : {elapsed_time:.2f} seconds")

torch.save(attacker_model.state_dict(), "surrogat_model.pt")
