import json
import re
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

def decoder(name, z):
    for encoder in ['utf-8', 'cp1252', 'iso-8859-1', 'utf-16', 'utf-32']:
        try:
            return z.read(name).decode(encoder)
        except:
            continue
    return z.read(name)

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['feed']

def write_json_file(file_path, result):
    with open(file_path, 'w') as file:
        json.dump({"result": result}, file)

def preprocess_text(feed):
    return [re.sub(r'\W+', ' ', article.lower()) for article in feed]

def analyze_tfidf(feed):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(feed)
    feature_array = tfidf_matrix.toarray()
    # Use Isolation Forest for anomaly detection
    clf = IsolationForest(random_state=0).fit(feature_array)
    scores = clf.decision_function(feature_array)
    return any(score < 0 for score in scores)  # Return True if any anomaly is detected

def detect_encoding_patterns(feed):
    patterns = {
        'base64': r'^(?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=)?$',
        'hexadecimal': r'^(?:[0-9a-fA-F]{2})+$',
        'morse_code': r'[\.\-\]+'
    }
    for article in feed:
        for pattern in patterns.values():
            try:
                if re.search(pattern, article):
                    return True
            except:
                continue
    return False

def detect_hidden_message(input_file, output_file):
    feed = read_json_file(input_file)
    preprocessed_feed = preprocess_text(feed)

    if analyze_tfidf(preprocessed_feed) or detect_encoding_patterns(preprocessed_feed):
        result = analyze_tfidf(preprocessed_feed) or detect_encoding_patterns(preprocessed_feed)
    else:
        result = analyze_tfidf(preprocessed_feed) or detect_encoding_patterns(preprocessed_feed)
    
    write_json_file(output_file, result)

# Example usage
input_file = 'example_feed.json'  # The file should contain a JSON object with a "feed" key
output_file = 'output.json'
detect_hidden_message(input_file, output_file)