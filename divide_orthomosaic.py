# Python libraries
import argparse

from os import listdir
from PIL import Image

# Local imports
from modules.managers.folder_manager import check_folder_existence
from modules.managers.image_manager import crop_large_images


def save_chunck(image_chunk:Image, output_dir: str):
    """
        This function receives a PIL Image object and save it in a directory

        :param image_chunck: PIL Image object
        :param output_dir: String Directory to save images
    """
    
    number = len(listdir(output_dir))
    image_chunk.save(f'{output_dir}/chunk_{number}.png', 'PNG')

def crop_and_save_large_images(input_path: str, output_dir: str, chunk_size: tuple=(256, 256)):
    """
        This function receives a path to a large image in .tif format, crop into small images, and save
        them into the ouput directory.

        :param input_path: String Path to original image .tif
        :param output_dir: String Directory to save all small images .png
        :param chunck_size: Tuple(int, int) The size of the new small images
    """
    # Check if the output directory exist
    check_folder_existence(output_dir)

    for image_chunk, *_ in crop_large_images(input_path, chunk_size):
        # Save the cunk
        save_chunck(image_chunk, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide orthomosaic into smaller sections.")
    
    # Define command-line arguments
    parser.add_argument("--input", type=str, required=True, help="Path to the input orthomosaic file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--chunk_size", type=tuple, required=False, help="Size of the output cropped image. Example: (256,256)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    crop_and_save_large_images(args.input, args.output)
