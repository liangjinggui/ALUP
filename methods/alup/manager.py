import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW


from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


from losses.contrastive_loss import SupConLoss 
from utils import view_generator, save_results, clustering_score, save_model
from .model import Bert
from .utils_data import MemoryBank, fill_memory_bank, NIDData, NeighborsDataset


class Manager:

    def __init__(self, args, data, pretrained_model=None, logger_name='Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')   
        self.logger.info(self.device)

        self.num_labels = data.num_labels
        self.logger.info('Number of known classes: %s', str(self.num_labels))

        self.model = Bert(args)
        self.model.to(self.device)
        self.tokenizer = self.model.tokenizer

        if pretrained_model is not None:
            self.load_pretrained_model(pretrained_model)

        self.best_model = None
        # training data
        self.prepare_data(args, data)

        # loss func
        self.cl_loss_fct = SupConLoss(temperature=0.07, contrast_mode='all',
                                      base_temperature=0.07)

        # optimizer and scheduler
        num_train_steps = int(len(self.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs
        self.optimizer, self.scheduler = self.get_optimizer(args, args.lr, num_train_steps)

        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)


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
            num_training_steps=num_steps
        )
        return optimizer, scheduler


    def load_pretrained_model(self, pretrained_model):
        """load the backbone of pretrained model"""
        self.logger.info('Loading pretrained model ...')
        pretrained_dict = pretrained_model.backbone.state_dict()
        self.model.backbone.load_state_dict(pretrained_dict, strict=False)


    def get_neighbor_dataset(self, args, data, indices):
        """convert indices to dataset"""
        dataset = NeighborsDataset(self.train_semi_dataset, indices)
        self.train_neighbor_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    
    def get_neighbor_inds(self, args, data):
        """get indices of neighbors"""
        memory_bank = MemoryBank(
            len(self.train_semi_dataset), 
            args.embed_feat_dim, len(data.all_label_list), 0.1)
        fill_memory_bank(self.train_semi_dataloader, self.model, memory_bank, self.device)
        indices = memory_bank.mine_nearest_neighbors(args.topk, calculate_accuracy=False, gpu_id=args.gpu_id)
        return indices


    def get_adjacency(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1 # if in neighbors
                if (targets[b1] == targets[b2]) and (targets[b1]>0) and (targets[b2]>0):
                    adj[b1][b2] = 1 # if same labels
                    # this is useful only when both have labels
        return adj


    def train(self, args, data): 

        best_model = None
        wait = 0
        best_metrics = {
            'Epoch': 0,
            'ACC': 0,
            'ARI': 0,
            'NMI': 0
        }
        self.logger.info("Start train ...")
        indices = self.get_neighbor_inds(args, data)
        self.get_neighbor_dataset(args, data, indices)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(self.train_neighbor_dataloader, desc="Iteration"):
                # 1. load data
                pos_neighbors = batch["possible_neighbors"] # all possible neighbor inds for anchor
                data_inds = batch["index"] # neighbor data ind

                # 2. get adjacency matrix
                adjacency = self.get_adjacency(args, data_inds, pos_neighbors, batch["target"]) # (bz,bz)

                # 3. obtaining different views
                anchor_input_ids = batch["anchor_input_ids"].to(self.device)
                anchor_attention_mask = batch["anchor_attention_mask"].to(self.device)
                neighbor_input_ids = batch["neighbor_input_ids"].to(self.device)
                neighbor_attention_mask = batch["neighbor_attention_mask"].to(self.device)
                anchor_input_ids = self.generator.random_token_replace(anchor_input_ids.cpu()).to(self.device)
                neighbor_input_ids = self.generator.random_token_replace(neighbor_input_ids.cpu()).to(self.device)

                # 4. compute loss and update parameters
                with torch.set_grad_enabled(True):

                    anchor_sent_embed = self.model(
                        input_ids=anchor_input_ids, 
                        attention_mask=anchor_attention_mask, 
                        mode='simple_forward'
                    )
                    neighbor_sent_embed = self.model(
                        input_ids=neighbor_input_ids, 
                        attention_mask=neighbor_attention_mask, 
                        mode='simple_forward'
                    )
                    
                    sent_embed = torch.stack([anchor_sent_embed, neighbor_sent_embed], dim=1)

                    loss = self.cl_loss_fct(sent_embed, mask=adjacency)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
                    tr_loss += loss.item()
                    nb_tr_examples += anchor_input_ids.size(0)
                    nb_tr_steps += 1

                    
            tr_loss = tr_loss / nb_tr_steps
            self.logger.info("***** Epoch: %s: Train loss: %f *****", str(epoch), tr_loss)
            results = self.test(args)

            if args.save_results:
                self.logger.info("***** Save results *****")
                results['epoch'] = epoch
                save_results(args, results)

            if results['ACC'] + results['ARI'] + results['NMI'] > \
                best_metrics['ACC'] + best_metrics['ARI'] + best_metrics['NMI']:
                best_metrics['Epoch'] = epoch
                best_metrics['ACC'] = results['ACC']
                best_metrics['ARI'] = results['ARI']
                best_metrics['NMI'] = results['NMI']
                self.best_model = copy.deepcopy(self.model.eval())
                save_model(args, self.best_model, 'best')
            
            self.model.eval()
            save_model(args, self.model, epoch)

            self.logger.info("***** Curr and Best model metrics *****")
            self.logger.info(
                "Curr Epoch %s Test Score: ACC = %s | ARI = %s | NMI = %s", 
                str(epoch), str(results['ACC']), str(results['ARI']), str(results['NMI'])
            )
            self.logger.info(
                "Best Epoch %s Test Score: ACC = %s | ARI = %s | NMI = %s", 
                str(best_metrics['Epoch']), str(best_metrics['ACC']), str(best_metrics['ARI']), str(best_metrics['NMI'])
            )

            # update neighbors every several epochs
            if ((epoch + 1) % args.update_per_epoch) == 0:
                indices = self.get_neighbor_inds(args, data)
                self.get_neighbor_dataset(args, data, indices)


    def test(self, args):
        
        feats, y_true = self.get_outputs(args, dataloader=self.test_dataloader, get_feats=True)
        km = KMeans(n_clusters = self.num_labels).fit(feats)
        y_pred = km.labels_
    
        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

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
                sent_embed = self.model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=None, 
                    mode='feature_ext')

                total_labels = torch.cat((total_labels, batch['label_id']))
                total_features = torch.cat((total_features, sent_embed))
                if not get_feats:  
                    logits = None
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

    def prepare_data(self, args, data):
        
        new_data = NIDData(args, data, self.tokenizer)

        self.train_semi_dataset = new_data.train_semi_dataset
        train_semi_sampler = SequentialSampler(self.train_semi_dataset)
        self.train_semi_dataloader = DataLoader(
            self.train_semi_dataset,
            sampler=train_semi_sampler,
            batch_size=args.train_batch_size
        )

        self.test_dataset = new_data.test_dataset
        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=args.eval_batch_size
        )