import cv2
import numpy as np
from PIL import Image


def crop_large_images(input_path: str, chunk_size: tuple=(256, 256)) -> "Generator[Image]":
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

                yield image_chunk, box, (width, height)

def check_image_shape(image: np.ndarray, avg_shape: tuple) -> np.ndarray:
    """
        Check if the image is smaller or bigger than the average shape. If smaller, will
        add black background. If bigger, will crop from the beginning and exclude the rest.

        :param image: A numpy array representing the image
        :param avg_shape: The average shape with 3 dimensions
        :return: The cropped/expanded image in a numpy array
    """
    height, width, _ = image.shape
    avg_height, avg_width, _ = avg_shape

    # Initialize extended_image with the original image
    extended_image = image.copy()

    # If the image is larger than the average shape, crop it
    if height > avg_height or width > avg_width:
        extended_image = extended_image[:avg_height, :avg_width, :]

        return extended_image

    # Check if the height is smaller than the average height
    if height < avg_height:
        # Create a new image with the desired height, maintaining the original width
        extended_image = np.zeros((avg_height, width, 3), dtype=np.uint8)
        vertical_offset = (avg_height - height) // 2
        extended_image[vertical_offset:vertical_offset + height, :, :] = image

    # Update height after possible extension
    height = extended_image.shape[0]

    # Check if the width is smaller than the average width
    if width < avg_width:
        # Create a new image with the desired width, maintaining the (possibly updated) height
        extended_image_with_width = np.zeros((height, avg_width, 3), dtype=np.uint8)
        horizontal_offset = (avg_width - width) // 2
        extended_image_with_width[:, horizontal_offset:horizontal_offset + width, :] = extended_image
        extended_image = extended_image_with_width

    return extended_image

def convert_to_grayscale(image):
    return np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), axis=-1)

def pil_to_cv2_image(pil_image):
    """
        Function to convert PIL Image objects into cv2 objects
    """
    # Convert the PIL image to a NumPy array
    cv2_image = np.array(pil_image)

    # Check the number of channels in the image
    if cv2_image.shape[-1] == 4:
        # Convert RGBA to RGB
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGBA2RGB)
    elif cv2_image.ndim == 2:  # Grayscale image
        # Convert grayscale to RGB
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB)
    return cv2_image

def show_image(image: cv2.imread):
    # Display the output image
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()