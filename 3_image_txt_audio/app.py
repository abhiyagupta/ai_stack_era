import os

# Install packages
os.system('pip install numpy')
os.system('pip install librosa')
os.system('pip install nltk')
os.system('pip install opencv-python')  # Install OpenCV
import nltk
#nltk.download('punkt')
nltk.download('punkt_tab')
import numpy as np
import librosa
from nltk.tokenize import word_tokenize


from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
from processors import (
    preprocess_text,
    preprocess_image,
    preprocess_audio,
    augment_text,
    augment_image,
    augment_audio
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {
    'txt': 'text',
    'png': 'image',
    'jpg': 'image',
    'jpeg': 'image',
    'wav': 'audio',
    'mp3': 'audio'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_text_file(filepath):
    """Helper function to read text files"""
    with open(filepath, 'r') as f:
        return f.read()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            file_type = ALLOWED_EXTENSIONS[filename.rsplit('.', 1)[1].lower()]
            
            try:
                # Process the file based on its type
                if file_type == 'text':
                    # Read the original text
                    original_text = read_text_file(filepath)
                    
                    processed_path, processed_details = preprocess_text(filepath)
                    processed_text = read_text_file(processed_path)
                    
                    augmented_path, augmented_details = augment_text(processed_path)
                    augmented_text = read_text_file(augmented_path)
                    
                    return render_template('result.html',
                                        original=filename,
                                        original_text=original_text,
                                        processed_text=processed_text,
                                        augmented_text=augmented_text,
                                        processed=os.path.basename(processed_path),
                                        augmented=os.path.basename(augmented_path),
                                        file_type=file_type,
                                        processed_details=processed_details,
                                        augmented_details=augmented_details)
                elif file_type == 'image':
                    processed_path, processed_details = preprocess_image(filepath)
                    augmented_path, augmented_details = augment_image(processed_path)
                
                elif file_type == 'audio':
                    processed_path, processed_details = preprocess_audio(filepath)
                    augmented_path, augmented_details = augment_audio(processed_path)
                
                return render_template('result.html',
                                    original=filename,
                                    processed=os.path.basename(processed_path),
                                    augmented=os.path.basename(augmented_path),
                                    file_type=file_type,
                                    processed_details=processed_details,
                                    augmented_details=augmented_details)
            except ValueError as e:
                return f"Error: {str(e)}"
    
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True) 