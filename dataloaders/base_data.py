import csv
import sys
import logging
import os
import random
import torch
import numpy as np

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

max_seq_lengths = {'mcid':21, 'clinc':30, 'stackoverflow':45, 'banking':55}

class BaseDataNew(object):

    def __init__(self, args):

        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.logger_name = args.logger_name
        args.max_seq_length = max_seq_lengths[args.dataset]
        
        # process labels
        self.all_label_list = self.get_labels(self.data_dir)
        self.all_label_list = [str(item) for item in self.all_label_list]

        # calculate the number of known classes
        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)

        # conduct the split of known and unknown classes
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.unknown_label_list = self.difference(self.all_label_list, self.known_label_list)

        # create examples
        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(args, mode='train', separate=True)
        self.eval_examples = self.get_examples(args, mode='eval', separate=False)
        self.test_examples = self.get_examples(args, mode='test', separate=False)
        self.test_known_examples, self.test_unknown_examples = self.get_examples(args, mode='test', separate=True)


    def get_examples(self, args, mode, separate=False):
        """
            args:
                mode: train, eval, test
                separate: whether to separate known and unknown classes
        """
        # read data
        examples = self.read_data(self.data_dir, mode)

        if mode == 'train':
            train_labels = np.array([example.label for example in examples])
            train_labeled_ids = []
            # sample labeled data with a given labeled ratio
            for label in self.known_label_list:
                num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
                pos = list(np.where(train_labels == label)[0])                
                train_labeled_ids.extend(random.sample(pos, num))
            
            # separate known and unknown data
            train_labeled_examples = []
            train_unlabeled_examples = []
            for idx, example in enumerate(examples):
                if idx in train_labeled_ids:
                    train_labeled_examples.append(example)
                else:
                    train_unlabeled_examples.append(example)
            return train_labeled_examples, train_unlabeled_examples

        elif mode == 'eval':
            eval_examples = []
            for example in examples:
                if example.label in self.known_label_list:
                    eval_examples.append(example)
            return eval_examples

        elif mode == 'test':
            if not separate:
                return examples
            else:
                test_labels = np.array([example.label for example in examples])
                test_labeled_ids = []
                for label in self.known_label_list:
                    num = len(test_labels[test_labels == label])
                    pos = list(np.where(test_labels == label)[0])                
                    test_labeled_ids.extend(random.sample(pos, num))
                
                test_known_examples = []
                test_unknown_examples = []
                for idx, example in enumerate(examples):
                    if idx in test_labeled_ids:
                        test_known_examples.append(example)
                    else:
                        test_unknown_examples.append(example)
                return test_known_examples, test_unknown_examples

        else:
            raise ValueError('mode must be train, eval or test')


    def read_data(self, data_dir, mode):
        """
            read data from data_dir
        """
        if mode == 'train':
            lines = self.read_tsv(os.path.join(data_dir, "train.tsv"))
            examples = self.create_examples(lines, "train")
            return examples
        elif mode == 'eval':
            lines = self.read_tsv(os.path.join(data_dir, "dev.tsv"))
            examples = self.create_examples(lines, "train")
            return examples
        elif mode == 'test':
            lines = self.read_tsv(os.path.join(data_dir, "test.tsv"))
            examples = self.create_examples(lines, "test")
            return examples
        else:
            raise NotImplementedError(f"Mode {mode} not found")
        

    def read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                line = [l.lower() for l in line]
                lines.append(line)
            return lines
        

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
        
            
    def get_labels(self, data_dir):
        """See base class."""
        docs = os.listdir(data_dir)
        if "train.tsv" in docs:
            import pandas as pd
            test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
            labels = [str(label).lower() for label in test['label']]
            labels = np.unique(np.array(labels))
        elif "dataset.json" in docs:
            with open(os.path.join(data_dir, "dataset.json"), 'r') as f:
                dataset = json.load(f)
                dataset = dataset[list(dataset.keys())[0]]
            labels = []
            for dom in dataset:
                for ind, data in enumerate(dataset[dom]):
                    label = data[1][0]
                    labels.append(str(label).lower())
            labels = np.unique(np.array(labels))
        return labels
    

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            label_id = label_map[example.label]

            features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id))
        return features


    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)  # For dialogue context
            else:
                tokens_b.pop()

    
    def difference(self, a, b):
        _b = set(b)
        return [item for item in a if item not in _b]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        self.configs:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """
        base features for a single training example
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_id):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

if __name__ == "__main__":
    
    config_path = '../methods/intent_generation/config.yaml'

    from utils import load_yaml_config
    from easydict import EasyDict
    configs = load_yaml_config(config_path)

    args = EasyDict(configs)

    base_data = BaseData(args)

    dataloader = base_data.eval_dataloader

    for idx, batch in enumerate(dataloader):
        print(batch)
        exit()

    






        