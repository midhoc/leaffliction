import os
import click
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt

from Augmentation import balance_subdirectories


SPLIT = 0.2
NUM_EPOCH = 1
BATCH_SIZE = 4
RESIZE = 256
NORMALIZE_MEAN_LST = [0.485, 0.456, 0.406]
NORMALIZE_STD_LST = [0.229, 0.224, 0.225]
MODEL_FILENAME = 'leaffliction_model.pth'
ZIP_FILENAME = 'save.zip'


def prepare_data(raw_dirpath):
    total_dirpath, train_dirpath, val_dirpath = make_processed_image_dirs(
        raw_dirpath
    )
    data_augmentation(raw_dirpath, total_dirpath)
    split_data(total_dirpath, train_dirpath, val_dirpath)
    return train_dirpath, val_dirpath


def make_processed_image_dirs(raw_dirpath):
    data_dirpath = Path(raw_dirpath).parent.absolute()
    processed_dirpath = os.path.join(data_dirpath, "processed")
    total_dirpath = os.path.join(processed_dirpath, "total")
    train_dirpath = os.path.join(processed_dirpath, "train")
    val_dirpath = os.path.join(processed_dirpath, "val")
    for dirpath in [total_dirpath, train_dirpath, val_dirpath]:
        if not os.path.isdir(dirpath):
            print(f"{dirpath} created.")
            os.makedirs(dirpath)
    return total_dirpath, train_dirpath, val_dirpath


def data_augmentation(raw_dirpath, total_dirpath):
    for class_dirname in os.listdir(raw_dirpath):
        class_dirpath = os.path.join(raw_dirpath, class_dirname)
        dest_class_dirpath = os.path.join(total_dirpath, class_dirname)
        shutil.copytree(class_dirpath, dest_class_dirpath)
    print(f"Raw data copied in {total_dirpath}.")
    balance_subdirectories(total_dirpath)
    print(f"Data augmented in {total_dirpath} :")
    for class_dirname in os.listdir(total_dirpath):
        class_dirpath = os.path.join(total_dirpath, class_dirname)
        nb_files = len(os.listdir(class_dirpath))
        print(f"{nb_files} images in {class_dirpath}.")


def split_data(total_dirpath, train_dirpath, val_dirpath):
    for class_dirname in os.listdir(total_dirpath):
        class_dirpath = os.path.join(total_dirpath, class_dirname)
        train_class_dirpath = os.path.join(train_dirpath, class_dirname)
        val_class_dirpath = os.path.join(val_dirpath, class_dirname)
        print((f"Split data from {class_dirpath} "
               + f"into {train_class_dirpath} and {val_class_dirpath} :"))
        for dirpath in [train_class_dirpath, val_class_dirpath]:
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
        images = os.listdir(class_dirpath)
        val_size = int(len(images) * SPLIT)
        val_images = random.sample(images, val_size)
        for image in tqdm(images):
            src_filepath = os.path.join(class_dirpath, image)
            if image in val_images:
                dest_filepath = os.path.join(val_class_dirpath, image)
            else:
                dest_filepath = os.path.join(train_class_dirpath, image)
            shutil.copyfile(src_filepath, dest_filepath)
    print("Train and Validation datasets completed.")


def get_dataloaders(train_dirpath, val_dirpath):
    data_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_MEAN_LST,
            std=NORMALIZE_STD_LST
        )
    ])
    train_dataset = ImageFolder(train_dirpath, data_transform)
    val_dataset = ImageFolder(val_dirpath, data_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )
    return train_dataloader, val_dataloader


def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def validation(val_dataloader, model, device, epoch, step):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print((f"Epoch [{epoch + 1}/{NUM_EPOCH}, Step {step} "
           + f"Validation Accuracy: {100 * accuracy:.2f}%"))
    model.train()
    return accuracy


def train_loop(device,
               train_dataloader,
               val_dataloader,
               model,
               criterion,
               optimizer):
    print("Training starting...")
    accuracy_lst = []
    step = 0
    for epoch in range(NUM_EPOCH):
        model.train()
        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if step % 600 == 0:
                accuracy_lst.append(
                    validation(val_dataloader, model, device, epoch, step)
                )
            step += 1
    validation(val_dataloader, model, device, epoch, step)
    torch.save(model.state_dict(), MODEL_FILENAME)
    sns.lineplot(x=range(len(accuracy_lst)), y=accuracy_lst)
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title("Evolution of accuracy on validation set")
    plt.savefig("training.png")
    plt.show()
    print("Training over.")


def save(train_dirpath, val_dirpath):
    with zipfile.ZipFile(ZIP_FILENAME, "w") as zipf:
        zipf.write(MODEL_FILENAME)
        for dirpath in [train_dirpath, val_dirpath]:
            base = os.path.basename(dirpath)
            for class_dirname in os.listdir(dirpath):
                class_dirpath = os.path.join(dirpath, class_dirname)
                for filename in os.listdir(class_dirpath):
                    filepath = os.path.join(class_dirpath, filename)
                    dst_filepath = os.path.join(base, class_dirname, filename)
                    zipf.write(filepath, arcname=dst_filepath)
    print(f"Model and augmented data saved at {ZIP_FILENAME}.")


def train(train_dirpath, val_dirpath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader = get_dataloaders(train_dirpath,
                                                       val_dirpath)
    model = get_model(
        num_classes=len(train_dataloader.dataset.classes)
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loop(
        device,
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
    )
    save(train_dirpath, val_dirpath)


@click.command()
@click.argument('image_dirpath', type=click.Path(exists=True))
def main(image_dirpath):
    train_dirpath, val_dirpath = prepare_data(image_dirpath)
    train(train_dirpath, val_dirpath)


if __name__ == "__main__":
    main()
