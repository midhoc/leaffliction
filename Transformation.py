import os
import cv2
from plantcv import plantcv as pcv
import click
import numpy as np
import matplotlib.pyplot as plt

@click.command()
@click.argument('image_path', type=click.Path(exists=True), required=False)
@click.option('-src', '--source_path', type=click.Path(exists=True), help='Source path to an image or a directory of images.')
@click.option('-dst', '--destination_path', type=click.Path(), help='Destination path to save transformed images when processing a directory.')
def extract_leaf_characteristics(image_path, source_path, destination_path):
    if image_path:
        # Process a single image and display transformations
        process_single_image(image_path)
    elif source_path:
        # Process all images in the directory and save transformations
        process_images_in_directory(source_path, destination_path)
    else:
        click.echo("Either provide an image path or use -src to specify a source directory.")

def process_single_image(image_path):
    # Load the original image using plantcv
    original_img = cv2.imread(image_path)

    # Create a single figure for all transformed images
    plt.figure(figsize=(18, 5))

    # Plot original image
    plt.subplot(1, 7, 1)
    plt.imshow(original_img)
    plt.title('Original')
    plt.axis('off')

    # Apply transformations and plot the results
    # transformations = ['gaussian_blur', 'mask', 'roi_objects', 'analyze_objects', 'pseudolandmarks', 'color_histogram']
    transformations = ['mask', 'apply_mask', 'skeleton', 'analyze_objects', 'pseudolandmarks', 'roi']
    mask = None
    for i, transformation in enumerate(transformations, start=2):
        transformed_image = apply_transformation(original_img, transformation, mask)
        if transformation == 'mask':
            mask = transformed_image
        plt.subplot(1, 7, i)
        plt.imshow(transformed_image)
        plt.title(transformation.capitalize())
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def process_images_in_directory(source_path, destination_path):
    # Create the destination directory if it doesn't exist
    if destination_path:
        os.makedirs(destination_path, exist_ok=True)

    # Process all images in the directory
    for filename in os.listdir(source_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_path, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_transformed.png"
            output_path = os.path.join(destination_path, output_filename) if destination_path else None

            process_single_image_and_save(image_path, output_path)

def process_single_image_and_save(image_path, output_path):
    # Load the original image using plantcv
    original_img = cv2.imread(image_path)

    # Apply transformations
    mask = None
    transformations = ['mask', 'apply_mask', 'skeleton', 'analyze_objects', 'pseudolandmarks', 'roi']
    for transformation in transformations:
        transformed_image = apply_transformation(original_img, transformation, mask)
        if transformation == 'mask':
            mask = transformed_image

        # Save the transformed image
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output_filename = f"{transformation}_{os.path.basename(image_path)}"
            output_filepath = os.path.join(output_path, output_filename)
            cv2.imwrite(output_filepath, transformed_image)

def apply_transformation(image, transformation, mask = None):
    # Apply the specified transformation using plantcv
    # if transformation == 'gaussian_blur':
    #     return pcv.gaussian_blur(image, ksize=(15, 15), sigma_x=0, sigma_y=0)

    if transformation == 'mask':
        return create_leaf_mask(image)
    
    elif transformation == 'apply_mask':
        return pcv.apply_mask(image, mask, 'WHITE')

    elif transformation == 'skeleton':
        skeleton = pcv.morphology.skeletonize(mask)
        
        # image_with_skeleton = pcv.visualize.pseudocolor(skeleton, mask=image, background='image')
        return skeleton
    
    elif transformation == 'analyze_objects':
        # Example: Analyze objects by finding contours
        shape_image = pcv.analyze.size(image, mask)
        return shape_image

    elif transformation == 'pseudolandmarks':
        return pseudo_landmarks(image, mask)

    elif transformation == 'color_histogram':
        analysis_image = pcv.analyze.color(image, mask, colorspaces="all", label="default")

        return analysis_image
    elif transformation == 'roi':
        img = image.copy()
        non_zero_pixels = np.where(mask != 0)
        x, y, w, h = np.min(non_zero_pixels[1]), np.min(non_zero_pixels[0]), np.max(non_zero_pixels[1]) - np.min(non_zero_pixels[1]), np.max(non_zero_pixels[0]) - np.min(non_zero_pixels[0])
        return cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    else:
        raise ValueError(f"Transformation {transformation} is not supported.")

def create_leaf_mask(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = (35, 50, 50)
    upper_green = (90, 255, 255)

    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_mask


def pseudo_landmarks(img, mask):
    image = img.copy()
    top, bottom, center = pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask)
    for point in np.round(top).astype(int):
        cv2.circle(image, point[0], radius=3, color=(0, 255, 0), thickness=-1)
        
    for point in np.round(bottom).astype(int):
        cv2.circle(image, point[0], radius=3, color=(255, 0, 0), thickness=-1)
        
    for point in np.round(center).astype(int):
        cv2.circle(image, point[0], radius=3, color=(0, 0, 255), thickness=-1)
    return image

if __name__ == "__main__":
    extract_leaf_characteristics()
