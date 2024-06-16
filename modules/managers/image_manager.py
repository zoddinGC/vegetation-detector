import numpy as np

def check_image_shape(image: np.ndarray, avg_shape: tuple) -> np.ndarray:
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
