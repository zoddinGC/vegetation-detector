"""
A primeira etapa consiste em quebrar uma imagem TIFF em partes menores. Na prática, os
ortomosaicos podem ser muito grandes, o que inviabilizaria a execução de uma inferência
de rede neural. Para isso, precisa-se quebrá-los em blocos (ou janelas) menores.
Para isso, implemente um script Python que divida o arquivo de imagem original e blocos de
imagens menores, salvando-os em arquivos PNG ou JPG. Ao final, espera-se que sejamos
capazes de executar o script da seguinte forma:

python divide_orthomosaic.py --input </path/to/orthomosaic.tif>
--output </path/to/output/dir/>
"""

import argparse

from os import listdir
from PIL import Image


def save_chunck(image_chunk:Image, output_dir: str):
    """
        This function receives a PIL Image object and save it in a directory

        :param image_chunck: PIL Image object
        :param output_dir: String Directory to save images
    """
    
    number = len(listdir(output_dir))
    image_chunk.save(f'{output_dir}/chunck_{number}.png', 'PNG')


def crop_large_images(input_path: str, output_dir: str, chunk_size: tuple = (1000, 1000)):
    """
        This function receives a path to a large image in .tif format, crop it, and transform
        it into small images .png

        :param input_path: String Path to original image .tif
        :param output_dir: String Directory to save all small images .png
        :param chunck_size: Tuple(int, int) The size of the new small images
    """

    with Image.open(input_path) as img:
        # Max image dimensions
        width, height = img.size

        for y in range(0, height, chunk_size[1]):
            for x in range(0, width, chunk_size[0]):

                # Get the minimum value (chunck or image dimensions)
                right = min(x + chunk_size[0], width)
                bottom = min(y + chunk_size[1], height)
                box = (x, y, right, bottom)

                # Crop the image to the box
                image_chunk = img.crop(box)

                # Check if the chunk size matches the expected size (debug)
                print(f"Cropped chunk box: {box}, actual size: {image_chunk.size}")

                # Process the image chunk
                save_chunck(image_chunk, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide orthomosaic into smaller sections.")
    
    # Define command-line arguments
    parser.add_argument("--input", type=str, required=True, help="Path to the input orthomosaic file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    crop_large_images(args.input, args.output)
