"""
A terceira parte do desafio consiste em implementar uma arquitetura de Rede Neural
Artificial para segmentação de imagens (pode adotar arquiteturas conhecidas também) e
treiná-la. Considere quebrar esta implementação em partes para melhor organização do
código.
Ao final, espera-se que sejamos capazes de executar o treinamento de um modelo de rede
neural através do seguinte comando:
python train_model.py --rgb </path/to/images/dir> --groundtruth
</path/to/segmented/dir/> --modelpath </path/to/model.h5>
"""

# Python libraries
import cv2
import numpy as np

from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from os import listdir
from scipy.stats import mode
from sklearn.model_selection import train_test_split

# Local imports
from modules.neural_network import create_model
from modules.managers.string_manager import extract_numbers
from modules.managers.image_manager import check_image_shape, show_image
from modules.managers.folder_manager import check_folder_existence


def __pre_process_images(images: list[cv2.imread], avg_shape: list) -> cv2.imread:
    images = [check_image_shape(x, avg_shape) for x in images]

    return np.array(images)

def load_images_from_directory(directory_path: str) -> tuple[cv2.imread]:
    # Create a list of all images presents on the directory
    images_list = listdir(directory_path)
    images_list.sort(key=extract_numbers)

    # Load all images within the directory
    images = [cv2.imread(directory_path.strip('/') + '/' + x) for x in images_list]

    return images

def train_model(original_images_dir: str, mask_images_dir: str, model_path: str, debug: bool = False):
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
    original_images = __pre_process_images(original_images, avg_shape)
    mask_images = __pre_process_images(mask_images, avg_shape)

    # Normalize the images from 0 to 1 (0 = black, 255 = white)
    original_images = original_images / 255.0
    mask_images = mask_images / 255.0

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(original_images, mask_images, test_size=0.2, random_state=42)
    print('complete.')

    # Create the input layer for the model (head)
    print(avg_shape)
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
    train_model(original_images_dir='data/cropped', mask_images_dir='data/binary', model_path='data/models/best_model.h5')
