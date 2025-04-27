import torch
import torch.optim as optim
import torch.nn as nn
import os
from config import epochs, learning_rate, train_data_path, test_data_path, save_path
from model import MyCustomModel
from dataset import UnicornImgDataset, unicornLoader

def my_descriptively_named_train_function(train_data_path, test_data_path, save_path):
    # Load the datasets
    train_dataset = UnicornImgDataset(train_data_path, train=True)
    test_dataset = UnicornImgDataset(test_data_path, train=False)

    # Create data loaders
    train_loader = unicornLoader(train_dataset, shuffle=True)
    test_loader = unicornLoader(test_dataset, shuffle=False)

    # Initialize the model, criterion, optimizer, and scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    model = MyCustomModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            test_loss = 0.0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

            print(f'Test Accuracy: {100 * test_correct / test_total:.2f}%, Test Loss: {test_loss / len(test_loader):.4f}')

    # Save model weights
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))  # Create the folder if it doesn't exist
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Final model saved to {save_path}")

if __name__ == "__main__":
    # This block will run when train.py is executed directly
    my_descriptively_named_train_function(train_data_path, test_data_path, save_path)
