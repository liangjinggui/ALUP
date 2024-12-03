import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

import pickle as pkl
import json
import os
import random
import numpy as np

max_seq_lengths = {'mcid':21, 'clinc':30, 'stackoverflow':45, 'banking':55}

class NIDDataset(Dataset):

    def __init__(self, examples_dict_list):
        self.examples_dict_list = examples_dict_list

    def __len__(self):
        return len(self.examples_dict_list)
    
    def __getitem__(self, idx):
        return self.examples_dict_list[idx]
    
class NIDData:

    def __init__(self, args, data, tokenizer):
        
        self.logger = logging.getLogger(args.logger_name)
        self.max_seq_len = max_seq_lengths[args.dataset]
        self.tokenizer = tokenizer

        self.all_label_list = data.all_label_list
        self.known_label_list = data.known_label_list

        self.all_label_map = {label: i for i, label in enumerate(self.all_label_list)}
        self.known_label_map = {label: i for i, label in enumerate(self.known_label_list)}
        self.train_labeled_examples = data.train_labeled_examples
        self.train_unlabeled_examples = data.train_unlabeled_examples

        if 'ealry_discovery_num' in args and args.ealry_discovery_num > 0:

            sampled_examples = []

            for intent_class in self.all_label_list:

                intent_examples = []
                for ex in data.train_unlabeled_examples:
                    if ex.label == intent_class:
                        intent_examples.append(ex)
                if intent_class in self.known_label_list:
                    sampled_examples += intent_examples
                else:
                    sampled_examples += random.sample(intent_examples, args.ealry_discovery_num)

            random.shuffle(sampled_examples)
            data.train_unlabeled_examples = sampled_examples

        self.train_labeled_ex_list = self.process_data(
            data.train_labeled_examples,
            self.known_label_map,
            mask_label=False,
            append_label=True
        )

        self.train_unlabeled_ex_list = self.process_data(
            data.train_unlabeled_examples,
            self.all_label_map,
            mask_label=True
        )

        self.train_semi_ex_list = self.train_labeled_ex_list + self.train_unlabeled_ex_list

        self.eval_ex_list = self.process_data(
            data.eval_examples,
            self.known_label_map,
            mask_label=False,
            append_label=True
        )

        self.test_ex_list = self.process_data(
            data.test_examples,
            self.all_label_map,
            mask_label=False
        )

        self.train_labeled_dataset = NIDDataset(self.train_labeled_ex_list)
        self.train_unlabeled_dataset = NIDDataset(self.train_unlabeled_ex_list)
        self.train_semi_dataset = NIDDataset(self.train_semi_ex_list)
        self.eval_dataset = NIDDataset(self.eval_ex_list)
        self.test_dataset = NIDDataset(self.test_ex_list)
        

    def process_data(self, examples, label_map, mask_label=False, append_label=False):
        examples_dict_list = []
        pad_token_id = self.tokenizer.pad_token_id
        for idx, example in enumerate(examples):

            input_text = example.text_a.strip('\n')
            label_text = example.label.strip('\n')
            assert label_text in label_map
            label_id = label_map[label_text] if not mask_label else -1
            label_id_true = label_map[label_text]

            input_text_features = self.tokenizer.batch_encode_plus(
                [input_text], max_length=self.max_seq_len, 
                padding='max_length', return_tensors="pt", truncation=True
            )

            input_ids = input_text_features['input_ids'].squeeze(0)
            attention_mask = (input_ids != pad_token_id).to(torch.float32)

            ex_dict = {
                'id': idx,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label_id': label_id,
                'label_id_true': label_id_true,
                'input_text': input_text,
                'label_text': label_text
            }
            examples_dict_list.append(ex_dict)
        return examples_dict_list



## from memory_bank.py
class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True, gpu_id=0):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        # if faiss.get_num_gpus() > 1:
        #     print("Using mutiple GPU resources ...")
        #     index = faiss.index_cpu_to_all_gpus(index)
        # elif faiss.get_num_gpus() == 1:
        #     print("Using single GPU resource ...")
        #     res = faiss.StandardGpuResources()
        #     index = faiss.index_cpu_to_gpu(res, 0, index)
        # else:
        #     print("Did not specified GPU sources ...")
        #     exit()
        print("Using GPU resources ...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)

        print("Adding features ...")
        index.add(features)
        print("Starting to search ...")
        distances, indices = index.search(features, topk+1) # Sample itself is included
        print(indices)
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, device):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        sent_embed = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'], 
                           labels=None, mode='feature_ext')
        
        memory_bank.update(sent_embed, batch['label_id'])
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))
    print('Fill Memory Done!')


## from neighbor_dataset.py
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()

        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])

        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)


        output['anchor_input_ids'] = anchor['input_ids']
        output['anchor_attention_mask'] = anchor['attention_mask']
        
        output['neighbor_input_ids'] = neighbor['input_ids']
        output['neighbor_attention_mask'] = neighbor['attention_mask']
        
        output['target'] = anchor['label_id']
        output['neighbor_target'] = neighbor['label_id']
        
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['index'] = index

        return output



