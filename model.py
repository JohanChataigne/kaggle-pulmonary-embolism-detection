import torch
import torch.optim as optim

def train(model, num_epoch, criterion, data_loader, optimizer, device):

    step_count = len(data_loader)

    print("Training begins")
    
    for epoch in range(num_epoch):
        for i, sample in enumerate(data_loader):

            image = sample['image'].to(device, dtype=torch.float)
            target = sample['target'].to(device, dtype=torch.float)
            target = target.view(target.shape[0], 1)

            # Reset gradiant
            optimizer.zero_grad()

            # Forward pass
            pred = model(image)

            # Compute loss
            loss = criterion(pred, target)

            # Backprop
            loss.backward()
            optimizer.step()

            # Debug
            if((i+1) % (step_count/10) == 0):
                print(f"Epoch [{epoch + 1}/{num_epoch}]" f", step [{i + 1}/{step_count}]" f", loss: {loss.item():.4f}")
                
    print("Finished training")
    
    return model
        
        
def evaluate(model, data_loader, device):
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        
        for sample in data_loader:
            
            images = sample['image'].to(device, dtype=torch.float)
            targets = sample['target'].to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%')