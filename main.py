import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from CNN-Architectures import LeNet_5

train_set = torchvision.datasets.MNIST(
   root="../../data/",
    train=True,
    download=True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

test_set = torchvision.datasets.MNIST(
    
    root = "../../data/",
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Device configuration: in our case we will use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LeNet_5().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

train_history = {'loss':[], 'accuracy':[]}
test_history = {'validation_loss':[], 'validation_accuracy':[]}

for epochs in range(100):
    # Training 
    model.train()
    total_loss = 0
    total_correct = 0
    total = 0
   
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predictions = outputs.max(1)
        total += labels.size(0)
        total_correct += predictions.eq(labels).sum().item()
        
    train_history['loss'].append(total_loss/len(train_set))
    train_history['accuracy'].append(total_correct/total)
    
    print(
        "Epoch: ", epochs,
        "Epoch loss: ", total_loss,
        "Accuracy: ", (total_correct/total)*100
    )
    
    # Validation...
    
    model.eval()
    total_test_loss = 0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            total_test_loss += loss.item()
            _, predictions = outputs.max(1)
            total += targets.size(0)
            total_correct += predictions.eq(targets).sum().item()
            
        test_history['validation_loss'].append(total_test_loss/len(test_set))
        test_history['validation_accuracy'].append(total_correct/total)
        
        
