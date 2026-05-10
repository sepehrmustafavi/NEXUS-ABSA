import torch
import logging
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class NeuroSymbolicDataset(Dataset):
    def __init__(self, dataset_name, data_split, tokenizer, max_len, sym_module, knowledge_type):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sym_module = sym_module
        self.knowledge_type = knowledge_type
        self.needs_token_type = 'token_type_ids' in tokenizer.model_input_names

        self.samples = []
        self._prepare_data(dataset_name, data_split)

    def _prepare_data(self, dataset_name, data_split):
        for item in data_split:
            if "mams" in dataset_name.lower():
                self.samples.append(item)
            elif "semeval" in dataset_name.lower():
                self.samples.append({
                    'text': str(item.get('text', '')),
                    'aspect': str(item.get('aspect', '')),
                    'label': int(item.get('sentiment', 0))
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        aspect = sample['aspect']
        label = sample['label']

        polarity = 0.0
        if self.knowledge_type == 'senticnet':
            polarity, _ = self.sym_module.get_text_polarity(text)
        elif self.knowledge_type == 'conceptnet':
            keywords = self.sym_module.get_keywords_from_text(text)
            scores = [self.sym_module.get_concept_score(k) for k in keywords]
            valid_scores = [s for s in scores if s != 0]
            polarity = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        encoding = self.tokenizer.encode_plus(
            text,
            aspect, 
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation='only_first', 
            return_token_type_ids=self.needs_token_type,
            return_attention_mask=True,
            return_tensors='pt',
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'symbolic_features': torch.tensor([polarity], dtype=torch.float),
            'raw_text': text,
            'aspect': aspect
        }
        
        if self.needs_token_type:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
            
        return item

def get_dataloaders(config, tokenizer, sym_module, knowledge_type, dataset_name):
    logging.info(f"Loading ABSA dataset: {dataset_name}...")
    import urllib.request
    import xml.etree.ElementTree as ET
    
    def parse_absa_xml(urls):
        data = []
        for url in urls:
            req = urllib.request.urlopen(url)
            tree = ET.parse(req)
            root = tree.getroot()
            for sentence in root.findall('sentence'):
                text = sentence.find('text').text
                aspect_terms = sentence.find('aspectTerms')
                if aspect_terms is not None and text is not None:
                    for aspect in aspect_terms.findall('aspectTerm'):
                        term = aspect.get('term')
                        polarity = aspect.get('polarity')
                        # حذف کلاس‌های conflict و متفرقه
                        if polarity in ['positive', 'negative', 'neutral']:
                            pol_int = 0 if polarity == 'negative' else 1 if polarity == 'neutral' else 2
                            data.append({'text': text, 'aspect': term, 'label': pol_int})
        return data

    if dataset_name == "mams":
        train_split = parse_absa_xml(["https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/train.xml"])
        val_split = parse_absa_xml(["https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/val.xml"])
        max_len = config.MAX_LEN_MAMS
        
    elif dataset_name == "semeval":
        train_split = parse_absa_xml([
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/restaurant_train.xml",
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/laptop_train.xml"
        ])
        val_split = parse_absa_xml([
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/restaurant_test.xml",
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/laptop_test.xml"
        ])
        max_len = config.MAX_LEN_SEMEVAL
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    train_ds = NeuroSymbolicDataset(dataset_name, train_split, tokenizer, max_len, sym_module, knowledge_type)
    val_ds = NeuroSymbolicDataset(dataset_name, val_split, tokenizer, max_len, sym_module, knowledge_type)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, max_len