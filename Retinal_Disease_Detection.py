import os
import time
import copy
import numpy as np

import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm import tqdm


def plot_loss_history(train_loss, val_loss):
    plt.figure(figsize=(20, 8))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, lw=3, color='red', label='Training Loss')
    plt.plot(epochs, val_loss, lw=3, color='green', label='Validation Loss')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training and Validation Loss', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.show()


class RetinalDisorderDataset(Dataset):
    def __init__(self, data_file, img_dir, transform=None):
        self.img_data = data_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_data.iloc[idx]['filename'])
        image = read_image(img_path)
        image = self.transform(image)
        label = self.img_data.iloc[[idx], 1:].values
        label = torch.tensor(label, dtype=torch.float32)
        return image, torch.squeeze(label)


# Training Function
def train_model(model, criterion, optimizer, scheduler, weights=None, num_epochs=25, model_name=None):
    model_name = model_name if model_name else model.__class__.__name__

    # Create the models folder (if not existing) to store the best model
    if not os.path.exists('models'): os.mkdir('models')

    since = time.time()
    # weights = weights.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    loss_history = {'train': [], 'val': []}

    for epoch in range(1, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # Set model to training or evaluation mode
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for idx, (inputs, labels) in tqdm(enumerate(dataloaders[phase]),
                                              leave=True,
                                              total=len(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs)
                    preds = torch.round(preds)

                    loss = criterion(outputs, labels)
                    # loss = (loss * weights).mean()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # running_loss += loss.item() * inputs.size(0)
                running_loss += loss.item()
                running_corrects += torch.sum((preds == labels.data).all(axis=1))

            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])
            loss_history[phase].append(epoch_loss)

            # print('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase.upper(), epoch_loss, epoch_acc))

            if phase == 'train':
                train_stats='{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase.upper(), epoch_loss, epoch_acc)
            else:
                print(train_stats + '{} Loss: {:.4f} Acc: {:.4f}'.format(phase.upper(), epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'models/' + model_name + '_v1.0.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_history


def get_pos_weight(df):
    pos_weight = []
    for c in range(df.shape[1]):
        weight = (df.iloc[:, c] == 0).sum() / (df.iloc[:, c] == 1).sum()
        pos_weight.append(weight)
    return pos_weight


if __name__ == '__main__':
    train_df = load_data('data/train/train.csv', ',')
    print(train_df.shape)

    # Let's visualize some rows of the data. The first columns corresponds to the
    # image file and the rest to whether the image contains any of the retinal diseases.
    train_df.head()

    # Examples of retinal images corresponding to each category.
    disease_labels = train_df.columns[1:]
    # Examples of retinal images corresponding to each category.
    for i in disease_labels:
      image_file = train_df.loc[train_df[i] == 1, 'filename'].sample().values[0]
      image = mpimg.imread('data/train/train/'+image_file)

      plt.title(i.upper())
      plt.axis("off")
      plt.imshow(image)
      plt.show()

    # Display the percentage and number of samples per disease label.
    category_percentage(train_df, disease_labels)

    plt.figure(figsize=(10, 5))
    train_df[disease_labels].sum().sort_values().plot(kind='barh')
    print(train_df[disease_labels].sum().sort_values())
    plt.show()

    # Correlation between disease.
    correlation_between_labels(train_df)

    # Now let's explore the interrelation between categories.
    venn_diagram(train_df, disease_labels, [0, 1, 3], [2, 4, 5], [1, 2, 3], [3, 5, 0])

    train_data, validation_data = train_test_split(train_df, train_size=0.90, random_state=42)
    print(train_data.shape)
    print(validation_data.shape)

    # Define image transforms
    img_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # DataFrame with train and validation info (i.e., filenames and labels)
    data_df = {'train': train_data, 'val': validation_data}

    # Create the dataset instances and dataloaders.
    image_dataset = {x: RetinalDisorderDataset(data_file=data_df[x],
                                               img_dir='data/train/train/',
                                               transform=img_transforms[x])
                     for x in ['train', 'val']}

    batch_size = 48
    dataloaders = {x: DataLoader(image_dataset[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    lr = 0.0001
    epochs = 30
    model = models.resnet50(pretrained=True, progress=True)
    # model_ft = models.densenet121(pretrained=True, progress=True)

    # ResNet50 has 4 layers, let's freeze the first three, and re-train only the last ones.
    for sub_layer in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]:
        for param in sub_layer.parameters():
            param.requires_grad = False

    # ct = 0
    # for child in model.children():
    #     ct += 1
    #     if ct < 8:
    #         for param in child.parameters():
    #             param.requires_grad = False


    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total number of parameters')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} parameters to train')

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(disease_labels))

    model = model.to(device)

    # pos_weight = get_label_weights()
    pos_weight = get_pos_weight(train_df.iloc[:, 1:])
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # weights=torch.tensor(pos_weight, dtype=torch.float32)
    model, loss_history = train_model(model, criterion, optimizer, exp_lr_scheduler,
                                      num_epochs=epochs, model_name='ResNet18')

    plot_loss_history(loss_history['train'], loss_history['val'])




