
### Twitter Sentiment and  Analysis Using Large Language Models (LLM)


### Project Overview
This project leverages Twitter data to perform sentiment analysis and topic generation using Large Language Models (LLMs). The focus is on tweets related to "attrition risk of employee, HR, or workplace" to provide insights into employee sentiments and prevalent topics of discussion. This model can be further adapted to analyze social media data related to other business or HR challenges.

### Components
Data Collection: Using Twitter API v2 to fetch tweets related to attrition risk, HR, or workplace topics.
Sentiment Analysis: Classifies tweets into different sentiment categories (e.g., positive, negative, neutral) using an LLM-based model.
Topic Generation: Uses an LLM to generate the primary topics of conversation from each tweet.
Visualization: Plots are generated to display sentiment distribution, top topics, and tweet frequency over time.

Dataset
The dataset consists of recent tweets fetched using Twitter API v2 based on the query: 'attrition risk of employee OR HR OR workplace lang:en'. Each tweet includes the following:

Timestamp: The time the tweet was posted.
Author_ID: Unique identifier for the author.
Tweet: The content of the tweet.

### Data Processing
Cleaning Tweets: Basic cleaning like removing special characters, links, and stop words.
Sentiment Analysis: Each cleaned tweet is analyzed by an LLM (e.g., transformers sentiment model) to classify sentiment.
We applied Random forest for validation and classification of the tweets according to the sentiments.
Topic Generation: Each cleaned tweet is processed to generate a topic using an LLM-based text generation model.
Model
Sentiment Analysis Model: We use a pre-trained transformer-based model (gpt2, or equivalent) for sentiment classification. The model takes in cleaned tweets and outputs a sentiment label (e.g., positive, negative, neutral).

Topic Generation Model: A generative LLM is used to identify the primary topics from the tweet. This helps in understanding the major themes and concerns related to the query terms.

Predicting skills of the employee according to the Job role which is based on the Job Description.
For example: If you search for Key word "Python Developer" then it will show the required skill for the same i.e Python.

Deployment of the model in the Flask 
We have 2 option in deployment, one more predicting skillset according to the job role and one is Generating topics according to the mentioned employee reviews.

### Results

Key Findings:
Sentiment Distribution: A clear visualization of the sentiment polarity (positive/negative/neutral) shows the general mood of tweets related to employee attrition and workplace.

Top Topics: The most frequently generated topics highlight the main themes being discussed, which could include keywords like "toxic workplace," "job satisfaction," "HR policies," etc.

Tweet Frequency: The tweet activity over time provides insights into when discussions are most active, helping target key time periods for interventions or further analysis.

Predicted skills: We can use the deployed models to fetch the skills required for the selected role or for the mentioned role.
