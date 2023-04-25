from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
#from wikiart_dataset import  Dataset_with_annotations
from torch.nn import Softmax
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset

# Use to optimize GPU usage
cudnn.benchmark = True
# For interactive mode
plt.ion()

def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs in dataloader:
                images = inputs['images'].to(device)
                labels = inputs['labels'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Visualize the predictions of the trained model
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


Resize = transforms.Compose([
    transforms.Resize((624, 624)),
])

#train_dataset =  Dataset_with_annotations('train')
#val_dataset =  Dataset_with_annotations('validation')
wikiart_dataset = load_dataset("huggan/wikiart")
wikiart_dataset.set_format(type="torch", columns=['image', 'artist', 'genre', 'style'])
train_dataset = Dataset({'images':Resize(wikiart_dataset['train']['image']), 'labels':wikiart_dataset['train']['genre']})
train_devtest = train_dataset.train_test_split(shuffle = True, seed = 32, test_size=0.2)
dataset = DatasetDict({
    'train': train_devtest['train'],
    'val': train_devtest['test']})
dataloader_train = DataLoader(dataset['train'], batch_size=16, shuffle=True, num_workers=0)
dataloader_val = DataLoader(dataset['val'] , batch_size=16, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the model (importing the pretrained weights and keeping the default activation function : ReLu)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
print('model uploaded')


# By default, the size of each output sample is set to 2 : (num_ftrs, 1) <==> Binary classification
# We changed nn.Linear(num_ftrs, 1) --> nn.Linear(num_ftrs, len(class_names)) = nn.Linear(num_ftrs, 27)
model_ft.fc = nn.Linear(num_ftrs, 27)

model_ft = model_ft.to(device)

# We define the loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer and learning rate
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# Decay learnimg rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, dataloader_train, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

visualize_model(model_ft)

if __name__ == '__main__':
    main()
