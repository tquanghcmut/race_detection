import os
from facenet_pytorch import MTCNN
import cv2
from PIL import Image

# Initialize the MTCNN detector
mtcnn = MTCNN()


def process_and_save_image(image_path):
    # Attempt to read the image
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        print(f"Error: Could not read the image {image_path}. It may be corrupt or not an image.")
        return

    # Convert the image to RGB
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL image
    image = Image.fromarray(image)

    # Detect the face and overwrite the original image
    try:
        face = mtcnn(image, save_path=image_path)
        if face is not None:
            print(f"Face detected and saved in {image_path}")
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")

# Main directory
main_dir = '/Users/twang/PycharmProjects/race_detection_official/test/input'

# Process images in the main directory
for image_name in os.listdir(main_dir):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(main_dir, image_name)
        process_and_save_image(image_path)
