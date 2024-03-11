import click
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from Train import RESIZE, NORMALIZE_MEAN_LST, NORMALIZE_STD_LST
from Test import load_model


def prediction(img_filepath, model, data_transform):
    idx2label_dct = {
        0: 'Apple_Black_rot',
        1: 'Apple_healthy',
        2: 'Apple_rust',
        3: 'Apple_scab',
        4: 'Grape_Black_rot',
        5: 'Grape_Esca',
        6: 'Grape_healthy',
        7: 'Grape_spot'
    }
    model.eval()
    with torch.no_grad():
        img = Image.open(img_filepath).convert('RGB')
        img = data_transform(img).unsqueeze(0)
        _, predicted_class = model(img).max(1)
        return idx2label_dct[predicted_class.item()]


def plot_predict(model, image_filepath, data_transform):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    img = Image.open(image_filepath).convert('RGB')
    ax0.imshow(img)
    transform = transforms.ToPILImage()
    img_tr = transform(data_transform(img))
    ax1.imshow(img_tr)
    fig.suptitle(("Prediction : "
                  + f"{prediction(image_filepath, model, data_transform)}"))
    plt.show()


@click.command()
@click.option('--model_filepath',
              '-m',
              type=click.Path(exists=True),
              required=True,
              help='model filepath')
@click.option('--image_filepath',
              '-i',
              type=click.Path(exists=True),
              required=True,
              help='image filepath.')
def main(model_filepath, image_filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_filepath, device)
    data_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_MEAN_LST,
            std=NORMALIZE_STD_LST
        )
    ])
    plot_predict(model, image_filepath, data_transform)


if __name__ == "__main__":
    main()
