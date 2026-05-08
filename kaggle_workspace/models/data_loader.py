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
                text = str(item.get('text', item.get('sentence', '')))
                aspect = str(item.get('aspect_term', ''))
                label = int(item.get('polarity', 0)) 

            elif "semeval" in dataset_name.lower():
                text = str(item.get('text', ''))
                aspect = str(item.get('aspect', ''))
                label = int(item.get('sentiment', 0))

            self.samples.append({
                'text': text,
                'aspect': aspect,
                'label': label
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
    
    if dataset_name == "mams":
        dataset = load_dataset("qiaojin/MAMS", "atsa")
        train_split = dataset['train']
        val_split = dataset['validation']
        max_len = config.MAX_LEN_MAMS
        
    elif dataset_name == "semeval":
        dataset = load_dataset("jakartaresearch/semeval-absa")
        train_split = dataset['train']
        val_split = dataset['validation'] if 'validation' in dataset else dataset['test']
        max_len = config.MAX_LEN_SEMEVAL
    
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    train_ds = NeuroSymbolicDataset(dataset_name, train_split, tokenizer, max_len, sym_module, knowledge_type)
    val_ds = NeuroSymbolicDataset(dataset_name, val_split, tokenizer, max_len, sym_module, knowledge_type)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, max_len