import os
import numpy as np
from PIL import Image
import librosa
import soundfile as sf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import cv2

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(filepath):
    """Preprocess text by removing stopwords and converting to lowercase"""
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    processed_tokens = [w for w in tokens if w not in stop_words]
    
    # Save processed text
    processed_path = os.path.join('processed', 'processed_' + os.path.basename(filepath))
    with open(processed_path, 'w') as f:
        f.write(' '.join(processed_tokens))
    
    details = {
        'original_words': len(tokens),
        'processed_words': len(processed_tokens),
        'stopwords_removed': len(tokens) - len(processed_tokens),
        'steps': [
            'Converted text to lowercase',
            f'Removed {len(tokens) - len(processed_tokens)} stopwords',
            'Tokenized the text'
        ]
    }
    
    return processed_path, details

def preprocess_image(filepath):
    """Preprocess image by converting to grayscale and applying histogram equalization"""
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    processed_path = os.path.join('processed', 'processed_' + os.path.basename(filepath))
    cv2.imwrite(processed_path, equalized)
    
    details = {
        'original_shape': img.shape,
        'processed_shape': equalized.shape,
        'steps': [
            'Converted image to grayscale',
            'Applied histogram equalization for better contrast',
            f'Original dimensions: {img.shape}',
            f'Processed dimensions: {equalized.shape}'
        ]
    }
    
    return processed_path, details

def preprocess_audio(filepath):
    """Preprocess audio by normalizing, removing silence, and slowing down"""
    y, sr = librosa.load(filepath)
    
    # Check audio length
    duration = len(y) / sr
    if duration > 60:
        raise ValueError("Audio file must be less than 1 minute long")
    
    # Normalize audio
    y_normalized = librosa.util.normalize(y)
    
    # Remove silence
    y_trimmed, indices = librosa.effects.trim(y_normalized, top_db=20)
    
    # Slow down the audio by 1.5x (reduce speed)
    y_slow = librosa.effects.time_stretch(y_trimmed, rate=1/1.5)
    
    processed_path = os.path.join('processed', 'processed_' + os.path.basename(filepath))
    sf.write(processed_path, y_slow, sr)
    
    details = {
        'original_duration': f"{len(y) / sr:.2f}",
        'processed_duration': f"{len(y_slow) / sr:.2f}",
        'sample_rate': sr,
        'silence_removed': f"{((len(y) - len(y_trimmed)) / sr):.2f}",
        'speed_factor': '1.5x slower',
        'steps': [
            'Checked audio duration (must be < 1 minute)',
            'Normalized audio amplitude',
            'Removed silence from beginning and end',
            'Slowed down audio by 1.5x',
            f'Original duration: {len(y) / sr:.2f} seconds',
            f'Processed duration: {len(y_slow) / sr:.2f} seconds',
            f'Removed {((len(y) - len(y_trimmed)) / sr):.2f} seconds of silence'
        ]
    }
    
    return processed_path, details

def augment_text(filepath):
    """Augment text by adding some basic synonyms"""
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Simple augmentation by duplicating some words
    augmented_text = text + " " + text
    
    augmented_path = os.path.join('processed', 'augmented_' + os.path.basename(filepath))
    with open(augmented_path, 'w') as f:
        f.write(augmented_text)
    
    details = {
        'original_length': len(text),
        'augmented_length': len(augmented_text),
        'steps': [
            'Duplicated the processed text',
            f'Original word count: {len(text.split())}',
            f'Augmented word count: {len(augmented_text.split())}'
        ]
    }
    
    return augmented_path, details

def augment_image(filepath):
    """Augment image by applying rotation and flipping"""
    img = cv2.imread(filepath)
    
    # Rotate image
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    
    # Flip image
    augmented = cv2.flip(rotated, 1)
    
    augmented_path = os.path.join('processed', 'augmented_' + os.path.basename(filepath))
    cv2.imwrite(augmented_path, augmented)
    
    details = {
        'steps': [
            'Rotated image by 45 degrees',
            'Applied horizontal flip',
            f'Image dimensions: {img.shape}',
            'Preserved original image size'
        ]
    }
    
    return augmented_path, details

def augment_audio(filepath):
    """Augment audio by adding noise and changing pitch"""
    y, sr = librosa.load(filepath)
    
    # Add noise
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    
    # Change pitch
    y_pitch = librosa.effects.pitch_shift(y_noise, sr=sr, n_steps=2)
    
    augmented_path = os.path.join('processed', 'augmented_' + os.path.basename(filepath))
    sf.write(augmented_path, y_pitch, sr)
    
    details = {
        'original_duration': f"{len(y) / sr:.2f}",
        'augmented_duration': f"{len(y_pitch) / sr:.2f}",
        'sample_rate': sr,
        'noise_amplitude': 0.005,
        'pitch_steps': 2,
        'steps': [
            'Added random noise (amplitude: 0.005)',
            'Shifted pitch up by 2 steps',
            f'Duration: {len(y) / sr:.2f} seconds',
            f'Sample rate: {sr} Hz'
        ]
    }
    
    return augmented_path, details