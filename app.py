from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)
# CORS zaroori hai taaki GitHub Pages ka frontend is backend API ko call kar sake
CORS(app) 

print("Model load ho raha hai... isme thoda time lag sakta hai.")
# Yahan "google/mt5-small" ki jagah baad mein apne Hugging Face model ka link daal sakte ho
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route('/')
def home():
    return "Hindi Question Generator API is Running!"

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        context = data.get('context', '')
        
        if not context:
            return jsonify({'error': 'No context provided'}), 400
            
        input_text = "generate question: " + context
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        outputs = model.generate(
            input_ids,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'question': question})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Cloud server ke liye port dynamically bind karna hota hai
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
