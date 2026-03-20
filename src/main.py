import json, re, base64, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
# from llama_cpp import Llama

# Basic decoder, not used for now
def decoder(name, z):
    for encoder in ['utf-8', 'cp1252', 'iso-8859-1', 'utf-16', 'utf-32']:
        try:
            return z.read(name).decode(encoder)
        except:
            continue
    return z.read(name)

# Will return True if text is not encoded in 'utf-8'
def read_json_file(file_path):
    global result
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data['feed']
    except:
        result = True

def write_json_file(file_path, result):
    with open(file_path, 'w') as file:
        json.dump({"result": result}, file)

def preprocess_text(feed):
    return [re.sub(r'\W+', ' ', article.lower()) for article in feed]

# Tfidf analysis
def analyze_tfidf(feed):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(feed)
    feature_array = tfidf_matrix.toarray()
    # Anomaly detection
    clf = IsolationForest(random_state=0).fit(feature_array)
    scores = clf.decision_function(feature_array)
    return any(score < 0 for score in scores)  # Only return True if any anomaly is detected

# Return True if any of this patterns are found inside the feed
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

def detect_hidden_message(input):
    feed = input['feed']
    preprocessed_feed = preprocess_text(feed)

    if analyze_tfidf(preprocessed_feed) or detect_encoding_patterns(preprocessed_feed):
        result = True
    else:
        result = True
    
    # print result(output, result)
    print(json.dumps({'result': result}))

input = json.load(sys.stdin)
# output = 'output.json'
detect_hidden_message(input)