import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_standard(model, train_loader, val_loader, epochs, 
                   lr=0.1, momentum=0.9, weight_decay=5e-4, 
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Standard Training loop.
    """
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Standard]")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)
            
            optimizer.zero_grad()
            outputs = model(x)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bs
            total_samples += bs
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()  # Moved after optimizer.step()
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        history['train_loss'].append(avg_loss)
        
        val_acc = _evaluate(model, val_loader, device)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Val Acc {val_acc:.4f}")

    return model, history

def _evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            if isinstance(outputs, tuple): outputs = outputs[0]
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return correct / total
