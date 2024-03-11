import os
import cv2
import click
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--balance', is_flag=True, help='Balance the number of images in each subdirectory using augmentation.')
def apply_augmentation(image_path, balance):
    if balance:
        balance_subdirectories(image_path)
    else:
        # Load the original image
        original_image = cv2.imread(image_path)
        image_name, image_extension = os.path.splitext(os.path.basename(image_path))

        # Define augmentation types
        augmentations = ['flip', 'rotate', 'skew', 'project', 'crop', 'brightness', 'translate', 'blur']

        # Create a single figure for all augmented images
        plt.figure(figsize=(18, 5))

        # Plot original image
        plt.subplot(1, 9, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')

        # Plot and save augmented images
        for i, augmentation_type in enumerate(augmentations, start=2):
            augmented_image = apply_augmentation_type(original_image.copy(), augmentation_type)
            augmented_image_path = f'{image_name}_{augmentation_type}{image_extension}'
            cv2.imwrite(augmented_image_path, augmented_image)

            plt.subplot(1, 9, i)
            plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
            plt.title(augmentation_type.capitalize())
            plt.axis('off')

        plt.tight_layout()
        plt.show()

def apply_augmentation_type(image, augmentation_type):
    if augmentation_type == 'flip':
        return cv2.flip(image, 1)  # Horizontal flip
    elif augmentation_type == 'rotate':
        angle = np.random.uniform(-30, 30)  # Random rotation angle between -30 and 30 degrees
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (cols, rows))
    elif augmentation_type == 'skew':
        shear_angle = np.random.uniform(-0.5, 0.5)  # Adjust the range as needed
        skew_matrix = np.float32([[1, shear_angle, 0], [0, 1, 0]])
        return cv2.warpAffine(image, skew_matrix, (image.shape[1], image.shape[0]))
    # elif augmentation_type == 'shear':
    #     shear_matrix = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
    #     return cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
    elif augmentation_type == 'crop':
        height, width = image.shape[:2]
        crop_size = int(min(height, width) * 0.8)
        start_x = np.random.randint(0, width - crop_size + 1)
        start_y = np.random.randint(0, height - crop_size + 1)
        return image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    elif augmentation_type == 'project':
        return apply_projection(image)
    elif augmentation_type == 'brightness':
        brightness_factor = np.random.uniform(0.4, 1.6)
        brightened_image = np.clip(image.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        return brightened_image
    elif augmentation_type == 'blur':
        kernel_size = np.random.choice(range(5, 16, 2))
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image
    elif augmentation_type == 'translate':
        height, width = image.shape[:2]
        fraction_x, fraction_y = np.random.uniform(-0.2, 0.2, size=2)
        translation_x = int(width * fraction_x)
        translation_y = int(height * fraction_y)
        translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        return cv2.warpAffine(image, translation_matrix, (width, height))

def apply_projection(image):
    width, height = image.shape[1], image.shape[0]
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    # Generate random perspective points
    random_margin = 0.2
    x_margin = width * random_margin
    y_margin = height * random_margin
    pts2 = np.float32([
        [np.random.uniform(0, x_margin), np.random.uniform(0, y_margin)],  # Top-left corner
        [np.random.uniform(width - x_margin, width), np.random.uniform(0, y_margin)],  # Top-right corner
        [np.random.uniform(0, x_margin), np.random.uniform(height - y_margin, height)],  # Bottom-left corner
        [np.random.uniform(width - x_margin, width), np.random.uniform(height - y_margin, height)]  # Bottom-right corner
    ])
    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, perspective_matrix, (width, height))

def balance_subdirectories(directory_path):
    max_images = 0
    for subdir, _, files in os.walk(directory_path):
        max_images = max(max_images, len(files))
    for subdir, _, files in os.walk(directory_path):
        if files:
            nb_files = len(files)
            nb_files_to_create = max_images - nb_files
            nb_created_files = 0
            i = 0
            while nb_created_files < nb_files_to_create:
                file_path = os.path.join(subdir, files[i % nb_files])
                original_image = cv2.imread(file_path)
                if original_image is not None:
                    image_name, image_extension = os.path.splitext(files[i % nb_files])
#                    augmentation_type = np.random.choice(['flip', 'rotate', 'skew', 'shear', 'crop', 'project', 'brightness', 'translate'])
                    augmentation_type = np.random.choice(['flip', 'rotate', 'skew', 'project', 'crop', 'brightness', 'translate', 'blur'])
                    augmented_image_path = os.path.join(subdir, f'{image_name}_{augmentation_type}{image_extension}')
                    if not os.path.isfile(augmented_image_path):
                        augmented_image = apply_augmentation_type(original_image.copy(), augmentation_type)
                        cv2.imwrite(augmented_image_path, augmented_image)
                        nb_created_files += 1
                i += 1

if __name__ == "__main__":
    apply_augmentation()
