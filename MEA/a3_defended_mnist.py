# -*- coding: utf-8 -*-



from __future__ import print_function

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR

from collections import defaultdict



class Lenet(nn.Module):

    def __init__(self):

        super(Lenet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)

        self.dropout1 = nn.Dropout(0.25)

        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(400, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = self.conv1(x)

        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.conv2(x)

        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.dropout2(x)

        x = self.fc2(x)

        x = F.relu(x)

        x = self.dropout2(x)

        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)

        return output
class DefendedModel:
    def __init__(self, base_model, max_queries_per_user=1000, round_precision=1):
        self.base_model = base_model
        self.base_model.eval()
        self.max_queries_per_user = max_queries_per_user
        self.round_precision = round_precision
        self.query_counters = defaultdict(int)  # Track number of queries per user

    def _round_softmax(self, output):
        softmax_probs = F.softmax(output, dim=1)
        rounded = torch.round(softmax_probs * (10 ** self.round_precision)) / (10 ** self.round_precision)
        return torch.log(rounded + 1e-8)  # Return log_probs for consistency

    def query(self, input_tensor, user_id="default"):
        if self.query_counters[user_id] >= self.max_queries_per_user:
            print(f"[BLOCKED] User '{user_id}' exceeded the query limit ({self.max_queries_per_user})")
            return None  # Or return torch.zeros_like(output) for stealth

        self.query_counters[user_id] += 1
        with torch.no_grad():
            raw_output = self.base_model(input_tensor)
            protected_output = self._round_softmax(raw_output)
            return protected_output

#Define normalization 

transform=transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.1307,), (0.3081,))

    ])

    

#Load dataset

dataset1 = datasets.MNIST('./data', train=True, download=True,

                   transform=transform)

dataset2 = datasets.MNIST('./data', train=False,

                   transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=True)



#Build the model we defined above

model = Lenet()



#Define the optimizer for model training

optimizer = optim.Adadelta(model.parameters(), lr=1)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)





model.train()

for epoch in range(1, 6):

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))

    scheduler.step()



model.eval()

test_loss = 0

correct = 0

with torch.no_grad():

    for data, target in test_loader:

        output = model(data)

        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        correct += pred.eq(target.view_as(pred)).sum().item()



test_loss /= len(test_loader.dataset)



print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

    test_loss, correct, len(test_loader.dataset),

    100. * correct / len(test_loader.dataset)))



torch.save(model.state_dict(), "defended_target_model.pt")