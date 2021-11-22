import cv2
import matplotlib.image as mpimg
from utils import *

import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class RetinalDisorderDataset(Dataset):
    def __init__(self, img_path, df, data):
        self.img_path = img_path
        self.df = df
        self.data = data
        self.image_names = list(self.df[:]['filename'])
        self.labels = list(np.array(self.df.drop(['filename'], axis=1)))

        if data == 'train':
            print(f"Number of training images: {len(self.df)}")
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # set the validation data images and labels
        elif data == 'val':
            print(f"Number of validation images: {len(self.df)}")
            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # set the test data images and labels, only last 10 images
        # this, we will use in a separate inference script
        elif data == 'test':
            # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(self.img_path + self.image_names[index])
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        return image, torch.tensor(targets, dtype=torch.float32)


# Training Function
def train_model(model, criterion, optimizer, scheduler, weights, num_epochs=25):
    since = time.time()
    weights = weights.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    train_loss = []
    validation_loss = []


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            run_Acc = 0.0
            run_Prec = 0.0
            run_Rec = 0.0
            run_F1 = 0.0

            # Iterate over data.
            for idx, (inputs, labels) in tqdm(enumerate(dataloaders[phase]),
                                              leave=True,
                                              total=int(len(image_datasets[phase])/dataloaders[phase].batch_size)):
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
                    loss = (loss * weights).mean()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum((preds == labels.data).all())

                if phase == 'val':
                    y_true, y_pred = labels.data.detach().cpu(), preds.detach().cpu()

                    run_Acc += accuracy_score(y_true, y_pred)
                    run_Prec += precision_score(y_true, y_pred, average="weighted", zero_division=0)
                    run_Rec += recall_score(y_true, y_pred, average="weighted", zero_division=0)
                    run_F1 += f1_score(y_true, y_pred, average="weighted", zero_division=0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = 100 * running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                # train_stats = '{} ==> Loss:{:.4f} Acc:{:.4f} Prec:{:.4f} Rec:{:.4f} F1:{:.4f}'.format(
                #     phase.upper(), epoch_loss, Acc_sc, Prec_sc, Rec_sc, f1_sc)
                train_stats = '{} ==> Loss:{:.4f}'.format(
                    phase.upper(), epoch_loss)
                train_loss.append(epoch_loss)
            else:
                Acc_sc = run_Acc / dataset_sizes[phase]
                Prec_sc = run_Prec / dataset_sizes[phase]
                Rec_sc = run_Rec / dataset_sizes[phase]
                f1_sc = run_F1 / dataset_sizes[phase]
                print(train_stats)
                print('{} ==> Loss:{:.4f} Acc:{:.4f} Prec:{:.4f} Rec:{:.4f} F1:{:.4f}'.format(
                    phase.upper(), epoch_loss, Acc_sc, Prec_sc, Rec_sc, f1_sc))
                validation_loss.append(epoch_loss)


            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            if phase == 'val' and epoch_loss < best_loss:
                # best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), PATH + 'models/MxResNet50_v1.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, validation_loss


def print_loss_history(train_loss, validation_loss, logscale=False):
    loss = train_loss
    val_loss = validation_loss
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.plot(epochs, val_loss, color='green', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if logscale:
        plt.yscale('log')
    plt.show()
    return


def get_label_weights():
  train_labels_df = pd.DataFrame(train_dataset.labels)
  pos_weight = []
  for c in range(train_labels_df.shape[1]):
    weight = len(train_labels_df) / (train_labels_df.iloc[:, c] == 1).sum()
    pos_weight.append(weight)
  return pos_weight


if __name__ == '__main__':
    PATH = ''
    train_df = load_data(PATH+'data/train/train.csv', ',')
    print(train_df.shape)

    # Let's visualize some rows of the data. The first columns corresponds to the
    # image file and the rest if the image contain any of the retinal disease.
    train_df.head(100)

    # Examples of retinal images correspinding to each category.
    disease_labels = train_df.columns[1:]

    for i in disease_labels:
      image_file = train_df.loc[train_df[i] == 1, 'filename'].sample().values[0]
      image = mpimg.imread(PATH+'data/train/train/'+image_file)

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

    train_data, validation_data = train_test_split(train_df, train_size=0.80, random_state=42, shuffle=True)
    print(train_data.shape)
    print(validation_data.shape)
    # Create the dataset instances and dataloaders.
    batch_size = 50
    # train dataset
    train_dataset = RetinalDisorderDataset(
        img_path=PATH+'data/train/train/', df=train_data, data='train'
    )
    # validation dataset
    valid_dataset = RetinalDisorderDataset(
        img_path=PATH+'data/train/train/', df=validation_data, data='val'
    )

    image_datasets = {
        'train': train_dataset,
        'val': valid_dataset
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=8)
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    lr = 0.0001
    epochs = 20
    model_ft = models.resnet50(pretrained=True, progress=True)
    # model_ft = models.densenet121(pretrained=True, progress=True)

    # Let's freeze the layers from 1 to 6. Then we would train only the remaining layers.
    ct = 0
    for child in model_ft.children():
        ct += 1
        if ct < 8:
            for param in child.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model_ft.parameters())
    print(f'{total_params:,} total number of parameters')
    total_trainable_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} parameters to train')

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(disease_labels))

    model_ft = model_ft.to(device)

    pos_weight = get_label_weights()
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft, train_loss, validation_loss = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               weights=torch.tensor(pos_weight, dtype=torch.float32),
                               num_epochs=epochs)

    print_loss_history(train_loss, validation_loss)




