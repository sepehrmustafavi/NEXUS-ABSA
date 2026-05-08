import os
import re
import json
import logging
import numpy as np
import nltk
from nltk.corpus import stopwords

# دانلود منابع پایه‌ای NLTK به صورت خاموش
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class SymbolicModuleSenticNet:
    """
    نسخه آفلاین و کش‌شده برای SenticNet.
    بدون نیاز به نصب کتابخانه سنگین senticnet، قطبیت کلمات را از فایل JSON می‌خواند.
    """
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logging.info(f"Loaded SenticNet cache with {len(data)} words.")
                    return data
            except Exception as e:
                logging.error(f"SenticNet cache load failed: {e}")
        else:
            logging.warning(f"SenticNet cache not found at {self.cache_path}! Polarity will be 0.0")
        return {}

    def get_text_polarity(self, text):
        """محاسبه میانگین قطبیت کلمات یافت‌شده در جمله"""
        if not self.cache:
            return 0.0, []
            
        # پیش‌پردازش متن
        text = re.sub(r'[^\w\s]', '', str(text))
        words = re.findall(r'\b\w+\b', text.lower())
        
        polarity_sum = 0.0
        found_keywords = []
        
        for word in words:
            # جستجوی کلمه در کش با سرعت O(1)
            if word in self.cache:
                item = self.cache[word]
                if item.get('found', False):
                    polarity_sum += item.get('score', 0.0)
                    found_keywords.append(word)
        
        count = len(found_keywords)
        if count == 0:
            return 0.0, []
        
        # کلیپ کردن میانگین قطبیت بین -1 و 1
        avg_polarity = np.clip(polarity_sum / count, -1.0, 1.0)
        return float(avg_polarity), list(set(found_keywords))


class ConceptNetModule:
    """
    نسخه کاملاً آفلاین ConceptNet.
    درخواست‌های API (requests.get) حذف شده‌اند تا از توقف آموزش در کگل جلوگیری شود.
    """
    def __init__(self, cache_path):
        self.cache_path = cache_path
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logging.info(f"Loaded ConceptNet cache with {len(data)} words.")
                    return data
            except Exception as e:
                logging.error(f"ConceptNet cache load failed: {e}")
        else:
            logging.warning(f"ConceptNet cache not found at {self.cache_path}!")
        return {}

    def get_keywords_from_text(self, text):
        """استخراج کلمات کلیدی با حذف Stop Words"""
        text = re.sub(r'[^\w\s]', '', str(text))
        try:
            words = nltk.word_tokenize(text.lower())
        except:
            words = text.lower().split()
            
        return list(set([w for w in words if w.isalpha() and w not in self.stop_words]))

    def get_concept_score(self, word):
        """بازگرداندن امتیاز کلمه از فایل کش (بدون Fallback به API)"""
        if word in self.cache:
            item = self.cache[word]
            if item.get('found', False):
                return float(item.get('score', 0.0))
        return 0.0