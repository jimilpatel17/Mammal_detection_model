from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from model_utils import MammalDetector, SpeciesClassifier
from utils import predict_mammal_with_gemini
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'media'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mammal_detector = MammalDetector()
species_classifier = SpeciesClassifier()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_jpeg(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    jpeg_path = image_path.rsplit('.', 1)[0] + '.jpg'
    img.save(jpeg_path, 'JPEG', quality=95)

    if jpeg_path != image_path:
        os.remove(image_path)

    return jpeg_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'prediction': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'prediction': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        jpeg_filepath = convert_to_jpeg(filepath)

        # Step 1: Check if the image contains a mammal
        mammal_result = mammal_detector.predict(jpeg_filepath, threshold=0.5)

        # Step 2: If not a mammal, return red text
        if not mammal_result['is_mammal']:
            return jsonify({
                'prediction':
                '<span style="color:#ffffff;">Not a Mammal</span>',
                'image_url':
                f'/media/{os.path.basename(jpeg_filepath)}'
            })


#if you want to use species classifier uncomment the following lines:

# # Step 3: Predict species name
# species_result = species_classifier.predict(jpeg_filepath, confidence_threshold=0.85)

# if species_result['is_confident']:
#     species_name = species_result['species']
# else:
#     gemini_result = predict_mammal_with_gemini(jpeg_filepath)
#     species_name = gemini_result['species']

# Step: Predict species name directly
        gemini_result = predict_mammal_with_gemini(jpeg_filepath)
        species_name = gemini_result['species']

        # Step 4: Return green text for valid mammal species
        return jsonify({
            'prediction':
            f'<span style="color:#ffffff;">{species_name}</span>',
            'image_url': f'/media/{os.path.basename(jpeg_filepath)}'
        })

    return jsonify({'prediction': 'Invalid file format'}), 400


@app.route('/media/<filename>')
def media_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
