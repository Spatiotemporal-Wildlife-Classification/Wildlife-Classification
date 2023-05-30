import tensorflow as tf
import os
import numpy as np
from src.structure.Config import root_dir
img_size = 528
model_name = 'family_taxon_classifier'

# Images to use for manual test
elephant_definite_01 = root_dir() + "/data/taxon/elephantidae/loxodonta/4321448_a.jpg"
elephant_definite_02 = root_dir() + "/data/taxon/elephantidae/loxodonta/9330730_a.jpg"
elephant_definite_03 = root_dir() + "/data/taxon/elephantidae/loxodonta/9619321_c.jpg"
elephant_definite_04 = root_dir() + "/data/taxon/elephantidae/loxodonta/loxodonta_africana/17097_a.jpg"

felid_definite_01 = root_dir() + "/data/taxon/felidae/felis/51075512_a.jpg"
felid_definite_02 = root_dir() + "/data/taxon/felidae/panthera/panthera_leo/22096_a.jpg"
felid_definite_03 = root_dir() + "/data/taxon/felidae/panthera/panthera_leo/2083512_b.jpg"


model_path = root_dir() + "/models/family_taxon_classifier"

if __name__ == "__main__":
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    image = tf.keras.utils.load_img(elephant_definite_04, target_size=(528, 528))
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)

    print(predictions)
