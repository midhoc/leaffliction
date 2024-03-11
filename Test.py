import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from Train import RESIZE, NORMALIZE_MEAN_LST, NORMALIZE_STD_LST, BATCH_SIZE


def get_dataloader(val_dirpath):
    data_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_MEAN_LST,
            std=NORMALIZE_STD_LST
        )
    ])
    val_dataset = ImageFolder(val_dirpath, data_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )
    return val_dataloader


def load_model(model_filepath, device):
    new_model = models.resnet18(weights=None)
    num_features = new_model.fc.in_features
    new_model.fc = nn.Linear(num_features, 8)
    new_model.load_state_dict(torch.load(model_filepath))
    new_model.to(device)
    return new_model


def test(model, val_dataloader, device):
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
    print(f"Validation Accuracy: {100 * accuracy:.2f}%")


@click.command()
@click.option('--model_filepath',
              '-m',
              type=click.Path(exists=True),
              required=True,
              help='model filepath')
@click.option('--val_dirpath',
              '-v',
              type=click.Path(exists=True),
              required=True,
              help='validation dirpath.')
def main(model_filepath, val_dirpath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_filepath, device)
    val_dataloader = get_dataloader(val_dirpath)
    test(model, val_dataloader, device)


if __name__ == "__main__":
    main()
