import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig

class PretrainBert(nn.Module):

    def __init__(self, args):
        super(PretrainBert, self).__init__()


        self.backbone = AutoModelForMaskedLM.from_pretrained(args.bert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        self.config = AutoConfig.from_pretrained(args.bert_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, args.known_num_labels)

    
    def forward(self, input_ids=None, attention_mask=None, 
                labels=None, mode='simple_forward'):
        if mode == 'simple_forward':
            assert labels is None
            sent_embed, outputs = self.simple_forward(input_ids=input_ids,
                                                      attention_mask=attention_mask)
            logits = self.classifier(sent_embed)
            return sent_embed, logits
        elif mode == 'mlm_forward':
            assert labels is not None
            sent_embed, outputs = self.simple_forward(input_ids=input_ids,
                                                      attention_mask=attention_mask, labels=labels)
            return outputs.loss
        elif mode == 'feature_ext':
            sent_embed, outputs = self.simple_forward(input_ids=input_ids,
                                                      attention_mask=attention_mask)
            logits = self.classifier(sent_embed)
            return sent_embed, logits
        else:
            raise ValueError('mode not supported')
    
    def simple_forward(self, input_ids=None,
                       attention_mask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, 
                                labels=labels, output_hidden_states = True)
        sent_embed = outputs.hidden_states[-1][:,0]
        sent_embed = self.dropout(sent_embed)
        return sent_embed, outputs



class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()

        self.backbone = AutoModelForMaskedLM.from_pretrained(args.bert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        self.config = AutoConfig.from_pretrained(args.bert_model)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, args.head_feat_dim)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, mode='feature_ext'):
        
        if mode=='feature_ext':
            sent_embed, outputs = self.simple_forward(input_ids=input_ids,
                                                      attention_mask=attention_mask)
            return sent_embed
        elif mode=='simple_forward':
            sent_embed, outputs = self.simple_forward(input_ids=input_ids,
                                                      attention_mask=attention_mask)
            sent_embed = F.normalize(self.head(sent_embed), dim=1)
            return sent_embed

    def simple_forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, 
                                labels=labels, output_hidden_states = True)
        sent_embed = outputs.hidden_states[-1][:,0]
        sent_embed = self.dropout(sent_embed)
        return sent_embed, outputs