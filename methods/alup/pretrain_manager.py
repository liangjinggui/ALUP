import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy

from tqdm import tqdm, trange
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from .model import PretrainBert
from utils import mask_tokens, accuracy_score, clustering_score
from .utils_data import NIDData
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class PretrainManager:

    def __init__(self, args, data, logger_name='Discovery'):

        self.logger = logging.getLogger(logger_name)
        self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')   

        args.known_num_labels = data.n_known_cls
        args.num_labels = data.num_labels
        self.logger.info('Number of known classes during pretrain: %s', str(args.known_num_labels))
        
        self.model = PretrainBert(args)
        self.model.to(self.device)
        self.tokenizer = self.model.tokenizer

        # training data
        self.prepare_data(args, data)

        # loss func
        self.ce_loss_fct = nn.CrossEntropyLoss()

        # optimizer and scheduler
        num_steps = int(len(self.train_labeled_dataset) / args.pretrain_batch_size) * args.num_pretrain_epochs
        self.optimizer, self.scheduler = self.get_optimizer(args, args.lr_pre, num_steps)

    def prepare_data(self, args, data):
        
        new_data = NIDData(args, data, self.tokenizer)

        self.train_labeled_dataset = new_data.train_labeled_dataset
        train_labeled_sampler = RandomSampler(self.train_labeled_dataset)
        self.train_labeled_dataloader = DataLoader(
            self.train_labeled_dataset, 
            sampler=train_labeled_sampler,
            batch_size=args.pretrain_batch_size
        )

        self.train_semi_dataset = new_data.train_semi_dataset
        train_semi_sampler = SequentialSampler(self.train_semi_dataset)
        self.train_semi_dataloader = DataLoader(
            self.train_semi_dataset,
            sampler=train_semi_sampler,
            batch_size=args.train_batch_size
        )

        self.eval_dataset = new_data.eval_dataset
        eval_sampler = SequentialSampler(self.eval_dataset)
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size
        )

        self.test_dataset = new_data.test_dataset
        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=args.eval_batch_size
        )

    
    def get_optimizer(self, args, lr, num_steps):
        num_warmup_steps = int(args.warmup_proportion*num_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps)
        return optimizer, scheduler

    def train(self, args):

        wait = 0
        best_model = None
        best_eval_score = 0

        semi_iter = iter(self.train_semi_dataloader) # mlm on semi-dataloader
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            
            self.model.train()

            tr_loss = 0
            tr_ce_loss, tr_mlm_loss = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.train_labeled_dataloader, desc="Iteration", leave=False)):
                batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                # load semi batch
                try:
                    semi_batch = next(semi_iter)
                    semi_batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in semi_batch.items()}
                except StopIteration:
                    semi_iter = iter(self.train_semi_dataloader)
                    semi_batch = next(semi_iter)
                    semi_batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in semi_batch.items()}

                # obtain masked input_ids
                mask_input_ids, mask_label_ids = mask_tokens(
                    semi_batch['input_ids'].cpu(), self.tokenizer, mlm_probability=0.15)
                mask_input_ids = mask_input_ids.to(self.device)
                mask_label_ids = mask_label_ids.to(self.device)

                # forward pass
                with torch.set_grad_enabled(True):
                    # simple forward
                    _, logits = self.model(input_ids=batch['input_ids'], 
                                           attention_mask=batch['attention_mask'], 
                                           labels=None, 
                                           mode='simple_forward')
                    ce_loss = self.ce_loss_fct(logits, batch['label_id'])

                    # mlm forward
                    mlm_loss = self.model(input_ids=mask_input_ids, 
                                          attention_mask=semi_batch['attention_mask'], 
                                          labels=mask_label_ids, 
                                          mode='mlm_forward')

                    loss = ce_loss + mlm_loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    tr_loss += loss.item()
                    tr_ce_loss += ce_loss.item()
                    tr_mlm_loss += mlm_loss.item()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += batch['input_ids'].size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            ce_loss = tr_ce_loss / nb_tr_steps
            mlm_loss = tr_mlm_loss / nb_tr_steps
            
            eval_y_true, eval_y_pred = self.get_outputs(args, dataloader=self.eval_dataloader, get_feats=False)
            eval_score = round(accuracy_score(eval_y_true, eval_y_pred) * 100, 2)

            eval_results = {
                'Train_loss': round(loss, 6),
                'Train_ce_loss': round(ce_loss, 6),
                'Train_mlm_loss': round(mlm_loss, 6),
                'Eval_score': eval_score,
                'Best_score':best_eval_score,
                'Wait_epoch': wait
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            # self.test(args)

            if eval_score > best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score
            elif eval_score > 0:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model

    
    def test(self, args):
        self.logger.info('Testing pretrained model...')
        feats, y_true = self.get_outputs(args, dataloader=self.test_dataloader, get_feats=True)
        km = KMeans(n_clusters = args.num_labels).fit(feats)
        y_pred = km.labels_
    
        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        self.logger.info("** Pretrained Test: Confusion Matrix **")
        self.logger.info("%s", str(cm))
        self.logger.info("** Pretrained Test results **")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))
        self.logger.info("***** Pretrained Test: Classification Report *****")

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results


    def get_outputs(self, args, dataloader, get_feats=False):

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,self.model.config.hidden_size)).to(self.device)
        total_logits = torch.empty((0, args.known_num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration", leave=False):

            batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.set_grad_enabled(False):
                sent_embed, logits = self.model(input_ids=batch['input_ids'], 
                                                attention_mask=batch['attention_mask'], 
                                                labels=None, 
                                                mode='feature_ext')

                total_labels = torch.cat((total_labels, batch['label_id']))
                total_features = torch.cat((total_features, sent_embed))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            return feats, y_true
        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            return y_true, y_pred
        
