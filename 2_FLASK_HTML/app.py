from flask import Flask, render_template, request, jsonify, url_for
import os

app = Flask(__name__)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve animal image based on selection
@app.route('/get_animal_image', methods=['POST'])
def get_animal_image():
    data = request.get_json()
    animal = data.get('animal')
    
    # Define paths to images stored locally
    image_paths = {
        'cat': 'images/cat.jpg',
        'dog': 'images/dog.jpg',
        'elephant': 'images/elephant.jpg'
    }
    
    image_path = image_paths.get(animal)
    
    if image_path and os.path.exists(os.path.join(app.static_folder, image_path)):
        image_url = url_for('static', filename=image_path)
        print(f"Image URL: {image_url}")  # Debug print
        return jsonify({'image_url': image_url}), 200
    else:
        print(f"Image not found for animal: {animal}")  # Debug print
        return jsonify({'error': 'Animal image not found'}), 404

# Route to handle file uploads
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file_info = {
        'file_name': file.filename,
        'file_size': file.content_length,
        'file_type': file.content_type
    }
    
    return jsonify(file_info), 200

if __name__ == '__main__':
    app.run(debug=True)