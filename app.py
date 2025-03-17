from flask import Flask, request, jsonify, render_template
from flask_pymongo import PyMongo
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import joblib
import re
import requests
import nltk
from sklearn.metrics import pairwise_distances_argmin_min

app = Flask(__name__)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/llm_database"
mongo = PyMongo(app)

# Load the LLM model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('model_dir/tokenizer')
    model = AutoModelForCausalLM.from_pretrained('model_dir/model')
except Exception as e:
    print(f"Error loading LLM models: {e}")

# Load skill prediction model and vectorizer
try:
    skill_model = joblib.load('model_dir/model/kmeans_model.joblib')
    vectorizer = joblib.load('model_dir/model/vectorizer.joblib')
except Exception as e:
    print(f"Error loading skill prediction models: {e}")

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define skills mapping for skill prediction
skills_mapping = [
    "Python", "Data Analysis", "Machine Learning", "Deep Learning", 
    "Data Visualization", "Statistics", "SQL", "Big Data", 
    "Cloud Computing", "Web Development"
]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def predict_skills(job_title, n_clusters=3):
    try:
        print(f"predict_skills called with job_title: {job_title}")
        
        # Preprocess and vectorize the job title
        processed_title = preprocess_text(job_title)
        vectorized_title = vectorizer.transform([processed_title])
        closest_clusters, distances = pairwise_distances_argmin_min(vectorized_title, skill_model.cluster_centers_)
        top_n_clusters = closest_clusters.argsort()[:n_clusters]

        # Map top N clusters to skill names
        predicted_skills = [skills_mapping[i] for i in top_n_clusters if i < len(skills_mapping)]
        print(f"Predicted Skills before keyword matching: {predicted_skills}")

        # Expanded keyword-based skill matching with additional job titles
        keyword_skills = {
            "python developer": ["Python", "Django", "Flask", "SQL", "REST APIs"],
            "data scientist": ["Python", "Data Analysis", "Machine Learning", "Deep Learning", "Statistics"],
            "data analyst": ["SQL", "Excel", "Data Visualization", "Power BI", "Tableau", "Python"],
            "machine learning engineer": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "Data Analysis"],
            "full stack developer": ["JavaScript", "React", "Node.js", "HTML", "CSS", "SQL", "REST APIs"],
            "web developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Git"],
            "software engineer": ["Java", "Python", "C++", "Algorithms", "Data Structures", "SQL"],
            "cloud engineer": ["AWS", "Azure", "GCP", "Cloud Computing", "Docker", "Kubernetes"],
            "devops engineer": ["AWS", "Docker", "Kubernetes", "CI/CD", "Jenkins", "Python", "Bash"],
            "business analyst": ["Excel", "SQL", "Data Analysis", "Power BI", "Communication", "Project Management"],
            "product manager": ["Agile Methodology", "Communication", "JIRA", "Roadmap Planning", "Stakeholder Management"],
            "ai engineer": ["Python", "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch"],
            "frontend developer": ["HTML", "CSS", "JavaScript", "React", "Vue.js", "Responsive Design"],
            "backend developer": ["Node.js", "Python", "Java", "SQL", "REST APIs", "Database Management"],
            "cybersecurity analyst": ["Network Security", "Ethical Hacking", "SIEM", "Incident Response", "Risk Assessment"],
            "data engineer": ["SQL", "Python", "ETL", "Data Warehousing", "Apache Spark", "Hadoop"],
            "ux designer": ["User Research", "Wireframing", "Prototyping", "Adobe XD", "UI/UX Design"],
            "financial analyst": ["Excel", "Financial Modeling", "Forecasting", "Data Analysis", "Accounting"],
            "network administrator": ["Networking", "Cisco", "Firewall Management", "VPN", "Troubleshooting"],
            "digital marketer": ["SEO", "Google Analytics", "Content Marketing", "Social Media", "PPC"],
            "hr manager": ["Recruiting", "Employee Relations", "Performance Management", "Payroll", "Onboarding"],
            "operations manager": ["Process Optimization", "Project Management", "KPI Tracking", "Budgeting"],
            "project manager": ["Project Planning", "Risk Management", "MS Project", "Scrum", "Stakeholder Management"],
            "qa engineer": ["Testing", "Automation", "Selenium", "JIRA", "Bug Tracking"]
            # Add more job titles and skills as needed
        }

        # Check for keywords in the job title and add relevant skills
        for keyword, skills in keyword_skills.items():
            if keyword in processed_title:
                predicted_skills.extend(skills)

        # Ensure no duplicate skills in the output
        predicted_skills = list(set(predicted_skills))
        print(f"Final Predicted Skills: {predicted_skills}")

        return predicted_skills if predicted_skills else ["No skills predicted."]
    
    except Exception as e:
        print(f"Error in predict_skills function: {e}")
        return ["Error predicting skills."]

def generate_topics(text, max_input_length=100, max_new_tokens=150, temperature=0.8, top_k=0, top_p=0.9):
    try:
        inputs = tokenizer.encode(text[:max_input_length], return_tensors='pt')
        outputs = model.generate(
            inputs, 
            max_new_tokens=max_new_tokens, 
            num_return_sequences=1, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p,
            do_sample=True,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in generating topics: {e}")
        return "Error generating topics."

# Fetch Google Jobs using SerpAPI
SERPAPI_KEY = "fe58e4bbc03b3c9fefafcc2afebb34c7d4464f7d335bad49cd7799dd6b5adb1b"  # Replace with your actual SerpAPI API key

def fetch_google_jobs(job_title, location):
    params = {
        "engine": "google_jobs",
        "q": f"{job_title} in {location}",
        "api_key": SERPAPI_KEY,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code == 200:
        data = response.json()
        jobs = data.get('jobs_results', [])
        print(f"Jobs fetched successfully: {jobs}")
        return jobs
    else:
        print(f"Error fetching jobs: {response.status_code} - {response.text}")
        return []

# Route to fetch job data
@app.route('/get_jobs', methods=['GET'])
def get_jobs():
    job_title = request.args.get('job_title', '')
    location = request.args.get('location', '')

    print(f"Received request for get_jobs with job_title: '{job_title}' and location: '{location}'")

    if not job_title or not location:
        print("Error: Missing job title or location")
        return jsonify({'error': 'Missing job title or location'}), 400

    # Fetch job data from SerpAPI
    jobs = fetch_google_jobs(job_title, location)
    job_counts = {}
    job_positions = {}

    for job in jobs:
        loc = job.get('location', 'Unknown')
        job_counts[loc] = job_counts.get(loc, 0) + 1
        job_positions.setdefault(loc, []).append(job.get('title', 'Unknown Position'))

    result = {
        "job_counts": job_counts,
        "job_positions": job_positions,
        "jobs": jobs
    }
    return jsonify(result)

@app.route('/process', methods=['POST'])
def process_text():
    try:
        data = request.json
        input_text = data.get('text', '')
        prediction_type = data.get('type', '')

        print(f"Received request for /process with type {prediction_type} and text '{input_text}'")

        if not input_text:
            print("Error: No input text provided")
            return jsonify({'error': 'No input text provided'}), 400

        if prediction_type == '1':
            generated_text = generate_topics(input_text)
        elif prediction_type == '2':
            generated_text = predict_skills(input_text)
        else:
            print("Error: Invalid prediction type")
            return jsonify({'error': 'Invalid prediction type'}), 400

        print(f"Generated output: {generated_text}")
        return jsonify({'processed_text': generated_text})
    
    except Exception as e:
        print(f"Error in /process route: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
