"""
Na segunda, você deverá gerar o dataset de segmentação de imagens, o qual irá alimentar
o modelo de rede neural na etapa de treinamento e validação.
Para isso, implemente um script Python que realize a binarização dos blocos das imagens,
de forma que atribua-se o valor 1 a todos os pixels correspondentes a algum tipo de
vegetação e 0 (zero) para os pixels que não sejam vegetação. Salve o resultado em
imagem de escala cinza (PNG ou JPG) em um diretório separado.

Ao final, espera-se que sejamos capazes de executar o script da seguinte forma:
python binarize_images.py --input </path/to/images/dir> --output
</path/to/segmented/dir/>
"""

import argparse
import numpy as np
import cv2

from os import listdir, makedirs
from sklearn.cluster import DBSCAN
from re import findall


def extract_numbers(input_string):
    # Find all sequences of digits in the string
    numbers = findall(r'\d+', input_string)
    return numbers[0]

def check_folder_existence(folder_path:str):
    """
    Check if there's a folder in the given folder path. If not, try to create.

    :param folder_path: Relative path to check the folder
    """
    try:
        try:
            # Check if folder exists
            listdir(folder_path)
        except:
            # Create of not
            makedirs(folder_path)
    except ValueError as e:
        print(f'Not possible to create folder in {folder_path}. Error: {e}')


def detect_green_pixels(image_path: str, output_path: str, key: int, debug: bool = False) -> list:
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if there's any 'label' folder in the output destination
    check_folder_existence(output_path)

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold values for green color
    lower_green = np.array([40, 0, 20])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Create the output image
    output_image = np.full_like(image, 255)  # Initialize with white

    # Set pixels in the green range to black
    output_image[mask == 255] = [0, 0, 0]  # Set to black where the mask is true

    # Get the plants clusters
    clusters = cluster_plants(mask, image)

    # Save the output image inverted (white = Plant)
    cv2.imwrite(output_path.strip('/') + f'/img_{key}.png', cv2.bitwise_not(output_image))

    if debug:
        # Change detect plants to black pixels
        image[mask == 255] = [0, 0, 0]

        # Display the output image
        cv2.imshow('Processed Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    write_clusters_yolo_notation(clusters, img_name=f'img_{key}', output_path=output_path)


def cluster_plants(mask, image):
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

def write_clusters_yolo_notation(clusters: list, img_name: str, output_path: str):
    # Create the labels path to save the clusters
    labels_path = output_path[:output_path.rfind('/')] + '/labels'

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
        detect_green_pixels(image_path, output_path=args.output, key=extract_numbers(image_path), debug=args.debug)
