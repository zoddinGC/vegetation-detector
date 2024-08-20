# Python libraries
import cv2
import numpy as np
import argparse

from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from os import listdir
from scipy.stats import mode
from sklearn.model_selection import train_test_split

# Local imports
from modules.neural_network import create_model
from modules.managers.string_manager import extract_numbers
from modules.managers.image_manager import check_image_shape, show_image, convert_to_grayscale
from modules.managers.folder_manager import check_folder_existence


def __pre_process_mask(masks: list[np.array]):
    """
        Function to convert all cv2 images within a list to a gray scale
    """
    return np.array([convert_to_grayscale(x) for x in masks])

def check_all_images_shape(images: list[np.array], avg_shape: list) -> np.array:
    """
        Function to check if all images inside of an array have the default shape of the model
    """
    images = [check_image_shape(x, avg_shape) for x in images]

    return np.array(images)

def load_images_from_directory(directory_path: str) -> tuple[cv2.imread]:
    """
        This function will load all images inside a directory.

        :param directory_path: A string to the directory containing all images
        :return: A tuple with all images in cv2.imread objects
    """
    # Create a list of all images presents on the directory
    images_list = listdir(directory_path)
    images_list.sort(key=extract_numbers)

    # Load all images within the directory
    return [cv2.imread(directory_path.rstrip('/') + '/' + x) for x in images_list]

def train_model(original_images_dir: str, mask_images_dir: str, model_path: str, debug: bool = False):
    """
        This function will load the images and masks to train a neural network model. The neural network architecture
        is customized using keras and tensorflow.

        :param original_images_dir: The directory containing all original and cropped images
        :param mask_images_dir: The directory of all masked images
        :param model_path: The path to save the best model
        :param debug: If True, will display the image on the user's screen
    """
    # Check the model path
    model_path = model_path if model_path.count('.h5') > 0 else model_path + '_.h5'
    check_folder_existence(model_path[:model_path.rfind('/')])
    print(f'Best model will be saved in {model_path!r}')

    # Load all images from the given directories
    print('Loading images...', end=' ')
    original_images = load_images_from_directory(original_images_dir)
    mask_images = load_images_from_directory(mask_images_dir)
    
    # Calculate the most common shape of the images
    avg_shape = mode(np.array([x.shape for x in original_images]))[0]
    print('complete.')

    print('Processing images...', end=' ')
    # Check with all images have the same shape
    original_images = check_all_images_shape(original_images, avg_shape)
    mask_images = check_all_images_shape(mask_images, avg_shape)

    # Convert mask to tensor format
    mask_images = __pre_process_mask(mask_images)
    print(f'Original shape: {original_images[0].shape} | Mask shape: {mask_images[0].shape}', end=' | ')

    # Normalize the images from 0 to 1 (0 = black, 255 = white)
    original_images = original_images / 255.0
    mask_images = mask_images / 255.0

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(original_images, mask_images, test_size=0.2, random_state=42)
    print('Complete.')

    # Create the input layer for the model (head)
    input_layer = Input(avg_shape)

    if debug:
        for original, binary in zip(original_images, mask_images):
            # Display the images
            show_image(original)
            show_image(binary)

    print('Loading model architecture and splitting the data...', end=' ')
    model = create_model(input_layer)
    print('complete.\n\n')

    if debug:
        model.summary()

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    print('==== Starting model training ====')    
    # Define the ModelCheckpoint callback to save the best model only
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min')

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, callbacks=[checkpoint])


if __name__ == '__main__':
    # debug
    # train_model(original_images_dir='data/cropped', mask_images_dir='data/binary', model_path='data/models/best_model.h5')

    parser = argparse.ArgumentParser(description="Train a neural network model to predict binary images.")
    
    # Define command-line arguments
    parser.add_argument("--rgb", type=str, required=True, help="Path to the all cropped images to train the model </path/to/images/dir>")
    parser.add_argument("--groundtruth", type=str, required=True, help="Path to the binary images </path/to/segmented/dir/>")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to save the best model </path/to/model.h5>")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    train_model(args.rgb, args.groundtruth, args.modelpath)
