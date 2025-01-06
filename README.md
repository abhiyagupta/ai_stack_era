# Punjabi BPE Tokenizer

This repository contains a Byte-Pair Encoding (BPE) tokenizer specifically designed for Punjabi text. 

## Features

- Custom BPE implementation for Punjabi
- Vocabulary size < 5000 tokens
- Compression ratio > 3.2X
- Gradio web interface for easy interaction
- Support for encoding and decoding Punjabi text

## Setup

1. Activate virtual environment
``` 
python -m venv venv 
.\venv\Scripts\Activate
```

2. Install requirements
```
pip install -r requirements.txt
```

3. Download and extract the punjabi language courpus from wiki dumps. 
![alt text](assets/downloading.png)
![alt text](assets/extracting.png)

```
python download_wiki_dumps.py
python extracting_dumps.py
```

4. Clean the extracted corpus
![alt text](assets/final_cleaning.png)

```
python clean_corpus.py
```

5. Make sure you have a Punjabi corpus file named `pa_corpus_cleaned.txt` in the root directory

6. Testing the tokenizer model locally 
![alt text](assets/size_before_after_training.png)

```
python model.py
```
7. Gradio interface
![alt text](assets/Gradio.png)

```
python app_gradio.py
```

## Usage

The Gradio interface will start and be accessible at `http://localhost:7860`

## Deployment

This app is also deployed to Hugging Face Spaces:

## TO UPDATE LATER 

## Note

The sample size parameter in the tokenizer has been optimized to achieve the required compression ratio of >3.2X. You can adjust this parameter in the `initialize_tokenizer()` function if needed.
