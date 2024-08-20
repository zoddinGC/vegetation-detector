# Python libraries
import argparse
import numpy as np
import cv2

from sklearn.cluster import DBSCAN
from os import listdir

# Local imports
from modules.managers.folder_manager import check_folder_existence
from modules.managers.string_manager import extract_numbers
from modules.managers.image_manager import show_image


def detect_green_pixels(image_path: str, output_dir: str, key: int, debug: bool = False) -> list:
    """
        This function will receive an image path, load the image and search for green pixels on it. After detecting those pixels,
        the function will save a new image with just white and black pixels in the output directory.

        :param image_path: Path to the image
        :param output_dir: Path to directory to save the image
        :param key: Image number to save
        :param debug: Boolean indicating if the image show be displayed on user's screen or not
    """

    # Read the image
    image = cv2.imread(image_path)
    
    # Check if there's any 'label' folder in the output destination
    check_folder_existence(output_dir)

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold values for green color
    lower_green = np.array([40, 0, 20])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Create the final image
    output_image = np.full_like(image, 255)

    # Set pixels in the green range to black
    output_image[mask == 255] = [0, 0, 0]

    # Get the plants clusters
    clusters = cluster_plants(mask, image)

    # Save the output image inverted (white = Plant)
    cv2.imwrite(output_dir.rstrip('/') + f'/img_{key}.png', cv2.bitwise_not(output_image))
    print(f'Image img_{key}.png processed')

    if debug:
        # Change detect plants to black pixels
        image[mask == 255] = [0, 0, 0]

        # Display the output image
        show_image(image)

    write_clusters_yolo_notation(clusters, img_name=f'img_{key}', output_dir=output_dir)


def cluster_plants(mask: cv2.imread, image: cv2.imread) -> list:
    """
        This function receives a HSV mask and the original image. Based on mask, it will create cluster of identified patterns,
        and merge them into one single cluster. This is useful to group similar elements such as plants.

        :param mask: cv2 object in HSV colors
        :param image: cv2 object original image to color
    """

    # Create an empty list to store all clusters
    clusters = []

    # Find green pixel coordinates
    green_pixels = np.column_stack(np.where(mask == 255))

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=1, min_samples=4).fit(green_pixels)
    labels = dbscan.labels_

    # Minimum cluster area to avoid noise
    min_area = 50

    for label in set(labels):
        if label == -1:
            continue  # Skip noise points

        # Extract cluster points
        mask = (labels == label)
        cluster_coords = green_pixels[mask]

        # Calculate the bounding box of the cluster
        x_min, y_min = np.min(cluster_coords, axis=0)
        x_max, y_max = np.max(cluster_coords, axis=0)

        # Calculate the area of the bounding box
        area = (x_max - x_min) * (y_max - y_min)
        
        if area < min_area:
            continue  # Skip small clusters

        # Draw a red rectangle around the cluster
        cv2.rectangle(image, (y_min, x_min), (y_max, x_max), (0, 0, 255), 2)

        # Create a YOLOv8 label notation format
        # <class_id> <x_center> <y_center> <width> <height>
        clusters.append([0, (x_min + x_max) // 2, (y_min + y_max) // 2, x_max - x_min, y_max - y_min])

    return clusters


def write_clusters_yolo_notation(clusters: list, img_name: str, output_dir: str):
    """
        This function receives a list of clusters and write a .txt in YOLOv8 notation.

        :param cluster: list with coordinates <class_id> <x_center> <y_center> <width> <height>
        :param img_name: str with the same na 741236me of the original image name
        :output_dir: str Directory to save the .txt file
    """
    # Create the labels path to save the clusters
    labels_path = output_dir[:output_dir.rfind('/')] + '/labels'

    # Check if there's any 'label' folder in the output destination
    check_folder_existence(labels_path)

    # Create a .txt file with the image name containing all labels coordinates
    with open(labels_path + f'/{img_name}.txt', 'w', encoding='utf-8') as file:
        for item in clusters:
            file.write(f"{' '.join(map(str, item))}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect plants on top images based on green color.")
    
    # Define command-line arguments
    parser.add_argument("--input", type=str, required=True, help="Path to directory containing all images.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--debug", type=bool, required=False, help="Image green detect displayed on screen.")
    
    # Parse arguments
    args = parser.parse_args()

    for image_path in listdir(args.input):
        image_path = args.input + '/' + image_path
        detect_green_pixels(image_path, output_dir=args.output, key=extract_numbers(image_path), debug=args.debug)
