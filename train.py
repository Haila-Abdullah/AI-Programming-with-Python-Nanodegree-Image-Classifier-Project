import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import os

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset.')
    parser.add_argument('data_directory', type=str)
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg13', 'vgg16'])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()

args = get_input_args()

# Data directories
train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'

# Define the training, validation, and testing sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build and train network
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, args.hidden_units)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(args.hidden_units, 128)),
    ('relu2', nn.ReLU()),
    ('output', nn.Linear(128, len(train_data.classes))),
    ('logsoftmax', nn.LogSoftmax(dim=1))
]))
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 50

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            val_loss = 0
            accuracy = 0
            num_batches = 0

            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)
                    val_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    num_batches += 1

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {val_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/num_batches:.3f}")

            running_loss = 0
            steps = 0
            model.train()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx
checkpoint = {
    'model_state_dict': model.state_dict(),
    'classifier': model.classifier,
    'class_to_idx': model.class_to_idx,
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': epochs
}

torch.save(checkpoint, os.path.join(args.save_dir, 'model_checkpoint.pth'))
