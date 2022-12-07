import io
import tensorflow as tf
from PIL import Image
import cityscapes
import numpy as np
from pathlib import Path

def get_segmentator():

    model = tf.keras.models.load_model('models/best_model.h5', custom_objects={'jaccard_loss':cityscapes.jaccard_loss, 'UpdatedMeanIoU':cityscapes.UpdatedMeanIoU})

    return model


def get_segments(model, binary_image, max_size=1024):

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    resized_image = input_image.resize((1024,1024))

    categories_img = Image.fromarray(
        cityscapes.cityscapes_category_ids_to_category_colors(
            np.squeeze(
                np.argmax(
                    model.predict(np.expand_dims(resized_image, 0)), axis=-1
                )
            )
        )
    )

    return categories_img
    
def get_segments_by_id(model, image_id):
    
    leftImg8bit_path = Path("data", "images")
    gtFine_path = Path("data", "labels")

    image_id = str(image_id)
    input_img_paths = []
    labels_img_paths = []
    if image_id:
        input_img_paths = sorted(
            Path(leftImg8bit_path).glob(f"*{image_id}_000019_leftImg8bit.png")
        )
        labels_img_paths = sorted(
            Path(gtFine_path).glob(f"*{image_id}_000019_color.png")
        )
    if (len(input_img_paths) == 0 or len(labels_img_paths) == 0):
        print("No image found!")
        return None, None
        
    with open(input_img_paths[0], "rb") as f:
        original_img_b64 = base64.b64encode(f.read())
        original_img_b64_str = original_img_b64.decode("utf-8")

        input_img = Image.open(
            BytesIO(base64.b64decode(original_img_b64))
        ).convert("RGB").resize(img_size)
        categories_img = Image.fromarray(
            cityscapes.cityscapes_category_ids_to_category_colors(
                np.squeeze(
                    np.argmax(
                        model.predict(np.expand_dims(input_img, 0)), axis=-1
                    )
                )
            )
        )
    
    with open(labels_img_paths[0], "rb") as f:
        labels_img_read = f.read()
        labels_img_b64 = base64.b64encode(labels_img_read)
        label_img = Image.open(labels_img_b64).convert("RGB")

    return categories_img, label_img