import io

import tensorflow as tf

from PIL import Image

import cityscapes

import numpy as np




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
                    model.predict(np.expand_dims(input_image, 0)), axis=-1
                )
            )
        )
    )

    return categories_img