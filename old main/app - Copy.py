from flask import Flask, request, jsonify, render_template
from flask_pymongo import PyMongo
from transformers import pipeline
import datetime
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel


app = Flask(__name__)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/llm_database"
mongo = PyMongo(app)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('model_dir/tokenizer')
model = AutoModelForCausalLM.from_pretrained('model_dir/model')

def generate_topics(text, max_input_length=50, max_new_tokens=20):
    # Truncate the input text if it's longer than the max_input_length
    inputs = tokenizer.encode(text[:max_input_length], return_tensors='pt')
    
    # Generate topics with a set number of new tokens
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Process the input text with the LLM and store results in MongoDB
@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    input_text = data['text']
    
    # Process the text using the LLM model
    generated_text = generate_topics(input_text)
    
    # Store the input and output in the MongoDB database
    record = {
        'input_text': input_text,
        'processed_text': generated_text,
        'timestamp': datetime.datetime.utcnow()
    }
    mongo.db.processed_texts.insert_one(record)

    # Return the processed text as JSON response
    return jsonify({'processed_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
