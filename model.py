import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def train(model, num_epoch, criterion, data_loader, optimizer, device, filename):

    step_count = len(data_loader)
    losses = list()

    print("Training begins")
    
    for epoch in range(num_epoch):
        
        epoch_loss = 0
        
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
            epoch_loss += loss
            # Backprop
            loss.backward()
            optimizer.step()

            # Debug
            if((i+1) % int(step_count/10) == 0):
                print(f"Epoch [{epoch + 1}/{num_epoch}]" f", step [{i + 1}/{step_count}]" f", loss: {loss.item():.4f}")
            
        losses.append(epoch_loss)
                
    print("Finished training")
    
    
    torch.save(model.state_dict(), filename)
    
    return model, losses
        
        
def evaluate(model, data_loader, device):
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sample in data_loader:

            images = sample['image'].to(device, dtype=torch.float)        
            targets = sample['target'].to(device)

            outputs = model(images)
            class_0 = torch.full((targets.size(0),1), 0.5).to(device)
            outputs = torch.cat((class_0, outputs), 1)

            _, pred = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (pred == targets).sum().item()
            
            
    return 100 * correct / total

    
    
def plot_loss(losses, num_epoch, step_count, filename):
    
    epochs = list(range(1, num_epoch+1))
    mean_losses = list(map(lambda x: x/step_count, losses))
    plt.plot(epochs, mean_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.grid()
    plt.savefig(filename)
    plt.show()