"""
A última etapa corresponde à fase da inferência, onde o modelo treinado será usado para
segmentar imagens RGB para identificação de regiões onde há vegetação ou não. Este
último algoritmo será usado para testes em imagens que não foram fornecidas para o
treinamento.
Considere buscar imagens capturadas por drones na internet para você mesmo avaliar o
nível de generalização de seu modelo treinado.
Ao final desta etapa, espera-se que sejamos capazes de executar a inferência do modelo
atraveś do seguinte comando:
python model_inference.py --rgb </path/to/image.png> --modelpath
</path/to/model.h5> --output </path/to/segmented/image.png>
"""

# Python libraries
from cv2 import imwrite
import argparse
import numpy as np
from keras.models import Model, load_model
from os import listdir

# Local imports
from modules.managers.image_manager import show_image, check_image_shape, pil_to_cv2_image, crop_large_images, remove_black_padding
from modules.managers.folder_manager import check_folder_existence, clean_folder


def load_model_from_path(model_path: str) -> Model:
    # Check misspeling characters from model path
    model_path = model_path.strip('/')
    model_path = model_path if model_path.count('.h5') > 0 else model_path + '.h5'

    return load_model(model_path)

def predict_images(larger_image_path: str, model_path: str, save_images_path: str, debug: bool = False):
    # Check if the given folder to save images exists
    check_folder_existence(save_images_path)
    clean_folder(save_images_path)

    # Load trained model
    model = load_model_from_path(model_path)
    chunk_size = model.input_shape[1:]

    for image, box, original_size in crop_large_images(input_path=larger_image_path, chunk_size=chunk_size[:2]):
        # Convert PIL object in cv2 object
        image = pil_to_cv2_image(image)

        # Check if the image is in the same shape of the trained model
        image = check_image_shape(image, chunk_size)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict image using trained model
        predict = model.predict(image)

        # Return image and predicted image to 3 dimensions
        image = np.squeeze(image, axis=0)
        predict = np.squeeze(predict, axis=0)

        # Save the predicted image
        imwrite(f'{save_images_path}/img_{len(listdir(save_images_path))}.png', predict * 255.0)

        # Show the image if debug is True
        if debug:
            show_image(image)
            show_image(predict)


if __name__ == "__main__":
    # predict_images(
    #     larger_image_path='data/original/Orthomosaico_roi.tif',
    #     model_path='data/models/best_model.h5',
    #     save_images_path='data/predict',
    #     debug=True
    # )

    parser = argparse.ArgumentParser(description="Predict binary images from a trained neural network model.")
    
    # Define command-line arguments
    parser.add_argument("--rgb", type=str, required=True, help="Path to a large image </path/to/images.png>")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to the trained model </path/to/model.h5>")
    parser.add_argument("--output", type=str, required=True, help="Path where the predicted image will be saved </path/to/segmendted/image.png>")
    parser.add_argument("--debug", type=bool, required=False, help="This option will show the original followed by predicted image.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    predict_images(args.rgb, args.modelpath, args.output, debug=args.debug)