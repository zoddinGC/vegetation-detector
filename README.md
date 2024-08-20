# Application purpose

This application is used to **crop larger images** into small ones (256x256 pixels), **identify vegetation** (green) from images, **train a neural network model**, and **predict vegetation** in new images.

**Example:**
- Cropped image (from a larger one)
![Original Cropped Image](https://raw.githubusercontent.com/zoddinGC/vegetation-detector/main/data/example/original_cropped_image.png)

- Binarized cropped image
![Binarized Cropped Image](https://raw.githubusercontent.com/zoddinGC/vegetation-detector/main/data/example/binarized_cropped_image.png)

# How to use
1. Clone this repository
`git clone https://github.com/zoddinGC/vegetation-detector/`
2. Install dependencies `pip install -r requirements.txt`
3. Insert original image in `data/original/[image_name].tif` (or .png)
4. Divide the image in small pieces of 256x256 pixels
	4.1 Run `python divide_orthomosaic.py --input </path/to/orthomosaic.tif> --output </path/to/output/dir/>`
5. Binarize the image (white pixels = vegetation)
	5.1 Run `python binarize_images.py --input </path/to/images/dir> --output </path/to/segmented/dir/>`
6. Train the model
	6.1 Run `python train_model.py --rgb </path/to/images/dir> --groundtruth </path/to/segmented/dir/> --modelpath </path/to/model.h5>`
7. Predict vegetation from new images
	7.1 Run `python model_inference.py --rgb </path/to/image.png> --modelpath </path/to/model.h5> --output </path/to/segmented/image.png>`

## Observation

Steps **4** to **6** are optional. There is a pre-trained model saved on `data/models/best_model.h5` for examplification.

You can simple use `python model_inference.py --rgb </path/to/image.png> --modelpath data/models/best_model.h5 --output </path/to/segmented/image.png>`