import re
import torch
import logging
import numpy as np
import torch.nn.functional as F

class XAIEngine:
    def __init__(self, device):
        self.device = device
        self.sbert = None

    def get_prob(self, model, tokenizer, text, aspect, symbolic_polarity, max_len, target_class):
        
        safe_text = text if text.strip() else "[UNK]"
        
        needs_token_type = 'token_type_ids' in tokenizer.model_input_names
        
        inputs = tokenizer.encode_plus(
            safe_text,
            aspect,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation='only_first', 
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=needs_token_type
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attn_mask = inputs['attention_mask'].to(self.device)
        sym_tensor = torch.tensor([[symbolic_polarity]], dtype=torch.float).to(self.device)
        
        kwargs = {}
        if needs_token_type:
            kwargs['token_type_ids'] = inputs['token_type_ids'].to(self.device)

        with torch.no_grad():
            outputs = model(input_ids, attn_mask, sym_tensor, **kwargs)
            probs = F.softmax(outputs, dim=1)
            return probs[0, target_class].item()

    def calculate_metrics(self, model, tokenizer, text, aspect, label, pred_class, keywords, sym_module, max_len):
        
        model.eval()
        
        polarity = 0.0
        if keywords:
            if hasattr(sym_module, 'get_text_polarity'):
                polarity, _ = sym_module.get_text_polarity(" ".join(keywords))
            elif hasattr(sym_module, 'get_concept_score'):
                scores = [sym_module.get_concept_score(k) for k in keywords]
                valid_scores = [s for s in scores if s != 0]
                polarity = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
             
        original_prob = self.get_prob(model, tokenizer, text, aspect, polarity, max_len, pred_class)

        # ---------------------------------------------------------
        # 1. Sufficiency
        # ---------------------------------------------------------
        text_sufficiency = " ".join(keywords) if keywords else ""
        suff_prob = self.get_prob(model, tokenizer, text_sufficiency, aspect, polarity, max_len, pred_class)
        sufficiency_score = (original_prob - suff_prob) ** 2

        # ---------------------------------------------------------
        # 2. Infidelity 
        # ---------------------------------------------------------
        text_infidelity = text
        for k in keywords:
            
            text_infidelity = re.sub(rf'\b{re.escape(k)}\b', "", text_infidelity, flags=re.IGNORECASE)
            
        inf_prob = self.get_prob(model, tokenizer, text_infidelity, aspect, 0.0, max_len, pred_class)
        infidelity_score = (original_prob - inf_prob) ** 2

        return {
            "sufficiency": sufficiency_score,
            "infidelity": infidelity_score,
            "keywords": keywords
        }