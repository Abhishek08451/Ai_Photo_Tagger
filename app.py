from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained AI model
try:
    model = tf.keras.models.load_model(r'C:\Users\abhis\Simple_Flask_App\model\savedmodel5.h5')
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise


@app.route('/', methods=['POST'])
def tag_photos():
    try:
        if 'photos' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        photos = request.files.getlist('photos')  # Get list of uploaded files
        tagged_photos = []

        for photo in photos:
            photo_data = preprocess_photo(photo)
            if photo_data is None:
                return jsonify({'error': 'Failed to preprocess photo'}), 500

            predictions = model.predict(np.expand_dims(photo_data, axis=0))
            tags = decode_predictions(predictions)
            tagged_photos.append({'photo': photo.filename, 'tags': tags})

        return jsonify(tagged_photos)

    except Exception as e:
        logging.error(f"Error in tag_photos: {e}")
        return jsonify({'error': str(e)}), 500


def preprocess_photo(photo):
    try:
        image = tf.image.decode_image(photo.read(), channels=3)
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0  # Normalize
        return image.numpy()
    except Exception as e:
        logging.error(f"Error preprocessing photo: {e}")
        return None

def decode_predictions(predictions):
    try:
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        tags = [{'class': str(i), 'confidence': str(predictions[0][i])} for i in top_indices]
        return tags
    except Exception as e:
        logging.error(f"Error decoding predictions: {e}")
        return []

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
