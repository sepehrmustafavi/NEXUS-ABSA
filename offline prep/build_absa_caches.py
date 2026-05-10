import os
import re
import sys
import time
import json
import logging
import requests
import numpy as np
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

CONCEPTNET_CACHE_PATH = "absa_conceptnet_cache.json"
SENTICNET_CACHE_PATH = "absa_senticnet_cache.json"

def get_unique_words_from_absa():
    stop_words = set(stopwords.words('english'))
    unique_words = set()
    import urllib.request
    import xml.etree.ElementTree as ET
    
    logging.info("Loading SemEval-2014 Dataset directly from XML...")
    try:
        semeval_urls = [
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/restaurant_train.xml",
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/restaurant_test.xml",
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/laptop_train.xml",
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/laptop_test.xml"
        ]
        for url in semeval_urls:
            req = urllib.request.urlopen(url)
            tree = ET.parse(req)
            root = tree.getroot()
            for sentence in root.findall('sentence'):
                text = sentence.find('text').text
                if text:
                    words = nltk.word_tokenize(re.sub(r'[^\w\s]', '', str(text).lower()))
                    unique_words.update([w for w in words if w.isalpha() and w not in stop_words])
    except Exception as e:
        logging.warning(f"Failed to load SemEval from XML: {e}")

    logging.info("Loading MAMS Dataset directly from XML...")
    try:
        mams_urls = [
            "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/train.xml",
            "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/val.xml",
            "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/test.xml"
        ]
        for url in mams_urls:
            req = urllib.request.urlopen(url)
            tree = ET.parse(req)
            root = tree.getroot()
            for sentence in root.findall('sentence'):
                text = sentence.find('text').text
                if text:
                    words = nltk.word_tokenize(re.sub(r'[^\w\s]', '', str(text).lower()))
                    unique_words.update([w for w in words if w.isalpha() and w not in stop_words])
    except Exception as e:
        logging.warning(f"Failed to load MAMS from XML: {e}")

    logging.info(f"Total unique words extracted from ABSA datasets: {len(unique_words)}")
    return list(unique_words)

def build_senticnet_cache(words):
    logging.info("Building SenticNet Cache...")
    try:
        from senticnet.senticnet import SenticNet
        sn = SenticNet()
    except ImportError:
        logging.error("Please install senticnet: pip install senticnet")
        return
        
    cache = {}
    for word in tqdm(words, desc="SenticNet Local Cache"):
        try:
            pol = float(sn.polarity_value(word))
            cache[word] = {"score": pol, "found": True}
        except:
            cache[word] = {"score": 0.0, "found": False}
            
    with open(SENTICNET_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
    logging.info(f"SenticNet Cache saved! ({len(cache)} words)")

def build_conceptnet_cache(words):
    logging.info("Building ConceptNet Cache (Requires API Calls)...")
    if os.path.exists(CONCEPTNET_CACHE_PATH):
        with open(CONCEPTNET_CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    words_to_fetch = [w for w in words if w not in cache]
    logging.info(f"Words remaining to fetch from ConceptNet: {len(words_to_fetch)}")

    api_url = "http://api.conceptnet.io/query"
    for i, word in enumerate(tqdm(words_to_fetch, desc="ConceptNet API Fetch")):
        try:
            response = requests.get(api_url, params={'node': f'/c/en/{word}', 'limit': 10}, timeout=5)
            score = 0.0
            found = False
            if response.status_code == 200:
                edges = response.json().get('edges', [])
                if edges:
                    weights = [e['weight'] for e in edges]
                    score = float(np.tanh(np.mean(weights) / 5.0))
                    found = True
            elif response.status_code == 429:
                time.sleep(5)
                continue
            cache[word] = {"score": score, "found": found}
        except:
            pass
        time.sleep(0.5)
        if (i + 1) % 100 == 0:
            with open(CONCEPTNET_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)

    with open(CONCEPTNET_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
    logging.info("ConceptNet Cache Build Complete!")

if __name__ == "__main__":
    logging.info("=== Starting Offline Data Preparation for NEXUS-ABSA ===")
    unique_vocab = get_unique_words_from_absa()
    if not unique_vocab:
        logging.error("No words extracted! Check internet connection.")
        sys.exit(1)
    build_senticnet_cache(unique_vocab)
    build_conceptnet_cache(unique_vocab)
    logging.info("=== ALL OFFLINE CACHES BUILT SUCCESSFULLY ===")