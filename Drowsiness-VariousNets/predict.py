from __future__ import print_function
from __future__ import division
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from PIL import Image
import time
import os
import copy
import datetime
import Vgg_face_dag

data_dir = "./data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "alexnet"

time_suffix = datetime.datetime.now().strftime('_%m%d_%H%M')

if not os.path.exists('model-' + model_name + time_suffix):
    os.mkdir('model-' + model_name + time_suffix)

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for
num_epochs = 50

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Detect if we have a GPU available
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

label_ids = {'normal', 'fatigue'}


def train_model(model,
                dataloaders,
                criterion,
                optimizer,
                scheduler,
                num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('lr: ', optimizer.state_dict()['param_groups'][0]['lr'])

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            confusion_matrix = np.zeros((num_classes, num_classes))

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for i in range(num_classes):
                    for j in range(num_classes):
                        confusion_matrix[i, j] += torch.sum((preds == j) *
                                                            (labels.data == i))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(
                    model.state_dict(), '{}/net-{:0>4d}.params'.format(
                        model_name, epoch))
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                print(confusion_matrix.astype('int'))
            if phase == 'train':
                scheduler.step(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Epoch: {:4f}'.format(best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # for param in model.parameters():
        #     param.requires_grad = False
        for name, param in model.named_parameters():
            param.requires_grad = False


def initialize_model(model_name,
                     num_classes,
                     feature_extract,
                     use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = (224, 224)

    if model_name == "image_flow_net":
        """ Image Flow Net
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = (224, 224)

    elif model_name == "vgg_face":
        """ VGG16
        """
        model_ft = Vgg_face_dag.vgg_face_dag(weights_path='params/vgg_face_dag.pth')
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc8.in_features
        model_ft.fc8 = nn.Linear(num_ftrs, num_classes)
        input_size = (224, 224)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name,
                                        num_classes,
                                        feature_extract,
                                        use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")


def open_and_pad(img_path):
    img = Image.open(img_path).convert('RGB')
    return img


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_transform):
        with open(data_path) as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            spans = line.strip().split()
            label = spans[0]
            base_path = data_path[:data_path.rfind('/') + 1]
            img_name = spans[1]
            sample = []

            img = open_and_pad(os.path.join(base_path, img_name))
            sample.append(img)

            sample.append(label_ids[label])
            self.samples.append(sample)

        self.len = len(self.samples)
        self.data_transform = data_transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, label = self.samples[index]
        img = self.data_transform(img)
        return img, label


# Create training and validation datasets
mydatasets = {
    x: MyDataset(os.path.join(data_dir, x + '_samples.txt'),
                 data_transforms[x])
    for x in ['train', 'val']
}

# Create training and validation dataloaders
dataloaders_dict = {
    'val':
    torch.utils.data.DataLoader(mydatasets['val'],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)
}

dataloaders_dict['train'] = torch.utils.data.DataLoader(mydatasets['train'],
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        pin_memory=True)

# Send the model to GPU
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9, weight_decay=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,
                                           mode='min',
                                           factor=0.1,
                                           patience=5,
                                           verbose=False,
                                           threshold=0.00001,
                                           threshold_mode='rel',
                                           cooldown=0,
                                           min_lr=0,
                                           eps=1e-08)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft,
                             dataloaders_dict,
                             criterion,
                             optimizer_ft,
                             scheduler,
                             num_epochs=num_epochs)
