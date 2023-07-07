"""
Library using tensorflow model to predict sequence data
"""

import numpy as np
import tensorflow as tf



def predict(model_file, weights_file, data):


    strategy = tf.distribute.MirroredStrategy(devices=None)


    with strategy.scope():

        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(weights_file)
        print(f"Model loaded from: {model_file} and {weights_file}")

        print("Final prediction")
        preds = model.predict(data)
        # FIXME strange workaround. why saluki model has 3 outputs. Whar are they?
        if len(np.shape(preds)) == 3:
            preds = preds[:,0,0]
        return(preds)

            