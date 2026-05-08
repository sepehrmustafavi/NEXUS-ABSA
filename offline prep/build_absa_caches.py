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
from datasets import load_dataset
from nltk.corpus import stopwords

# تلاش برای وارد کردن کتابخانه SenticNet
try:
    from senticnet.senticnet import SenticNet
    sn = SenticNet()
except ImportError:
    print("Please install senticnet: pip install senticnet")
    sys.exit(1)

# دانلود منابع NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# تنظیمات لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# مسیر فایل‌های خروجی
CONCEPTNET_CACHE_PATH = "absa_conceptnet_cache.json"
SENTICNET_CACHE_PATH = "absa_senticnet_cache.json"

def get_unique_words_from_absa():
    """خواندن دیتاست‌های ABSA و استخراج کلمات یکتا (بدون Stopwords)"""
    stop_words = set(stopwords.words('english'))
    unique_words = set()
    
    # 1. بارگذاری دیتاسِت SemEval-2014 Task 4 (Laptop & Restaurant)
    # از یکی از نسخه‌های معتبر Hugging Face استفاده می‌کنیم
    logging.info("Loading SemEval-2014 Dataset...")
    try:
        semeval = load_dataset("jakartaresearch/semeval-absa")
        for split in semeval.keys():
            for item in semeval[split]:
                text = item.get('text', '')
                words = nltk.word_tokenize(re.sub(r'[^\w\s]', '', str(text).lower()))
                unique_words.update([w for w in words if w.isalpha() and w not in stop_words])
    except Exception as e:
        logging.warning(f"Failed to load SemEval: {e}")

    # 2. بارگذاری دیتاسِت MAMS
    logging.info("Loading MAMS Dataset...")
    try:
        # نسخه MAMS در HF
        mams = load_dataset("qiaojin/MAMS", "atsa") 
        for split in mams.keys():
            for item in mams[split]:
                # در MAMS معمولا متن در فیلد text یا sentence است
                text = item.get('text', item.get('sentence', ''))
                words = nltk.word_tokenize(re.sub(r'[^\w\s]', '', str(text).lower()))
                unique_words.update([w for w in words if w.isalpha() and w not in stop_words])
    except Exception as e:
        logging.warning(f"Failed to load MAMS: {e}")

    logging.info(f"Total unique words extracted from ABSA datasets: {len(unique_words)}")
    return list(unique_words)

def build_senticnet_cache(words):
    """ساخت کش محلی برای SenticNet تا نیازی به نصب کتابخانه سنگین در کگل نباشد"""
    logging.info("Building SenticNet Cache...")
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
    """کوئری زدن به ConceptNet با رعایت Rate Limit و ساخت فایل کش"""
    logging.info("Building ConceptNet Cache (Requires API Calls)...")
    
    # لود کردن کش قبلی در صورت وجود (برای Resume شدن در صورت قطعی)
    if os.path.exists(CONCEPTNET_CACHE_PATH):
        with open(CONCEPTNET_CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        logging.info(f"Loaded existing cache with {len(cache)} words.")
    else:
        cache = {}

    words_to_fetch = [w for w in words if w not in cache]
    logging.info(f"Words remaining to fetch from ConceptNet: {len(words_to_fetch)}")

    api_url = "http://api.conceptnet.io/query"
    save_interval = 100
    
    for i, word in enumerate(tqdm(words_to_fetch, desc="ConceptNet API Fetch")):
        try:
            response = requests.get(api_url, params={'node': f'/c/en/{word}', 'limit': 10}, timeout=5)
            score = 0.0
            found = False
            
            if response.status_code == 200:
                edges = response.json().get('edges', [])
                if edges:
                    weights = [e['weight'] for e in edges]
                    raw_score = np.mean(weights)
                    score = float(np.tanh(raw_score / 5.0)) # نرمال‌سازی Tanh
                    found = True
            elif response.status_code == 429:
                logging.warning("Rate limit hit! Sleeping for 5 seconds...")
                time.sleep(5)
                continue # این کلمه در دور بعدی (ایندکس نشدن) دوباره بررسی نمیشود، پس باید ذخیره نشود
                
            cache[word] = {"score": score, "found": found}
            
        except Exception as e:
            # در صورت خطا (مثل تایم‌اوت)، ذخیره نمی‌کنیم تا بعدا دوباره تلاش شود
            pass
            
        # کانسپت‌نت محدودیت ۱ درخواست در ثانیه دارد (یک مقدار تاخیر برای اطمینان)
        time.sleep(0.5)
        
        if (i + 1) % save_interval == 0:
            with open(CONCEPTNET_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)

    # ذخیره نهایی
    with open(CONCEPTNET_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
    logging.info("ConceptNet Cache Build Complete!")

if __name__ == "__main__":
    logging.info("=== Starting Offline Data Preparation for NEXUS-ABSA ===")
    unique_vocab = get_unique_words_from_absa()
    
    if not unique_vocab:
        logging.error("No words extracted! Check dataset names/internet connection.")
        sys.exit(1)
        
    build_senticnet_cache(unique_vocab)
    build_conceptnet_cache(unique_vocab)
    logging.info("=== ALL OFFLINE CACHES BUILT SUCCESSFULLY ===")