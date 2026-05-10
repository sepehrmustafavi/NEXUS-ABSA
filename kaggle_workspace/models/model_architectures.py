import torch
import logging
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class NeuroSymbolicBase(nn.Module):
    def __init__(self, model_name, n_classes=3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        if hasattr(self.config, 'hidden_size'):
            self.hidden_dim = self.config.hidden_size
        elif hasattr(self.config, 'n_embd'): 
            self.hidden_dim = self.config.n_embd
        else:
            self.hidden_dim = 768 
            
        self.classifier = nn.Linear(self.hidden_dim + 1, n_classes)

    def forward_head(self, neural_features, symbolic_features):
        combined = torch.cat([neural_features, symbolic_features], dim=1)
        return self.classifier(combined)

class NeuroSymbolicEncoder(NeuroSymbolicBase):
    def __init__(self, model_name, n_classes=3):
        super().__init__(model_name, n_classes)
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, symbolic_features, token_type_ids=None, **kwargs):
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None and 'token_type_ids' in self.backbone.forward.__code__.co_varnames:
            inputs['token_type_ids'] = token_type_ids

        outputs = self.backbone(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.forward_head(cls_output, symbolic_features)


class NeuroSymbolicCausalLM(NeuroSymbolicBase):
    def __init__(self, model_name, n_classes=3):
        super().__init__(model_name, n_classes)
        self.backbone = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

    def forward(self, input_ids, attention_mask, symbolic_features, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
        else:
            seq_lengths = torch.full((batch_size,), input_ids.shape[1] - 1, device=input_ids.device)
            
        seq_lengths = torch.clamp(seq_lengths, min=0, max=input_ids.shape[1]-1)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        
        last_token_features = hidden_states[batch_indices, seq_lengths, :]
        return self.forward_head(last_token_features, symbolic_features)


def apply_tada(model, arch_type, strategy):
    logging.info(f"Configuring TADA: {strategy.upper()} for Architecture: {arch_type}")
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    backbone = model.backbone
    input_embeddings = backbone.get_input_embeddings()
    if input_embeddings:
        for param in input_embeddings.parameters():
            param.requires_grad = True

    if strategy == 'flexible':
        last_layer = None
        try:
            if arch_type in ['roberta', 'deberta']:
                last_layer = backbone.encoder.layer[-1]
            elif arch_type == 'qwen2':
                if hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
                    last_layer = backbone.model.layers[-1]
                elif hasattr(backbone, 'layers'):
                    last_layer = backbone.layers[-1]
            
            if last_layer:
                for param in last_layer.parameters():
                    param.requires_grad = True
                logging.info("-> TADA Flexible Mode: Last Transformer Layer Unfrozen.")
            else:
                logging.warning(f"Could not identify the last layer for {arch_type}")
        except Exception as e:
            logging.error(f"Error unfreezing last layer: {e}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Parameter Efficiency: Trainable Params {trainable:,} ({trainable/total:.2%})")