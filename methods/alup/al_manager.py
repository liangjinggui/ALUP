import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import faiss
import json
import os
import re
import copy

from torch.optim import AdamW
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import t
from scipy.optimize import linear_sum_assignment
from transformers import get_linear_schedule_with_warmup

from methods.alup.model import Bert
from methods.alup.utils_data import NIDData, NIDDataset, MemoryBank, fill_memory_bank, NeighborsDataset
from methods.alup.utils_prompt import PROMPT_BANKING, PROMPT_CLINC, PROMPT_STACKOVERFLOW
from methods.alup.utils_LLM import chat_completion_with_backoff, OPENAI_PIRCE
from losses.contrastive_loss import SupConLoss
from utils import clustering_score, view_generator, save_results, clustering_accuracy_score


class ALManager:

    def __init__(self, args, data, model_path, logger_name='Discovery'):


        assert model_path is not None, 'Model is None'
        
        self.logger = logging.getLogger(logger_name)
        self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')   
        self.logger.info(self.device)

        self.num_labels = data.num_labels
        self.n_known_cls = data.n_known_cls
        self.logger.info('Number of known classes: %s', str(self.n_known_cls))

        self.model = Bert(args)
        self.model.to(self.device)
        self.tokenizer = self.model.tokenizer
        
        self.load_pretrained_model(model_path)
        self.centroids = None

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


    def load_pretrained_model(self, model_path):
        
        if isinstance(model_path, str):
            self.logger.info('Loading pretrained model from %s', model_path)
            model_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model_dict['model_state_dict'])
        elif isinstance(model_path, Bert):
            self.logger.info('Loading pretrained model from a Bert object')
            self.model.load_state_dict(model_path.state_dict())
        else:
            raise ValueError('model_path should be a str or a Bert object')


    def get_neighbor_dataset(self, args, dataset, indices):
        """convert indices to dataset"""
        dataset = NeighborsDataset(dataset, indices)
        self.train_neighbor_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    
    def get_neighbor_inds(self, args, dataset, dataloader):
        """get indices of neighbors"""
        memory_bank = MemoryBank(
            len(dataset), 
            args.embed_feat_dim, len(self.all_label_list), 0.1)
        fill_memory_bank(dataloader, self.model, memory_bank, self.device)
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
    

    def prepare_data(self, args, data):
        
        new_data = NIDData(args, data, self.tokenizer)

        self.all_label_list = new_data.all_label_list

        self.train_semi_dataset = new_data.train_semi_dataset
        train_semi_sampler = SequentialSampler(self.train_semi_dataset)
        self.train_semi_dataloader = DataLoader(
            self.train_semi_dataset,
            sampler=train_semi_sampler,
            batch_size=args.train_batch_size
        )
        
        # llm_labels：用于存储LLM标注的label; llm_ex_indices: 用于选取LLM标注和label传播的样本
        self.llm_labels = np.array([ex['label_id'] for ex in new_data.train_semi_dataset])
        self.labeled_ex_labels = np.array([ex['label_id'] for ex in new_data.train_labeled_dataset])

        self.train_labeled_dataset = new_data.train_labeled_dataset
        self.train_labeled_examples = new_data.train_labeled_examples
        self.train_labeled_ex_list = new_data.train_labeled_ex_list
        train_labeled_sampler = SequentialSampler(self.train_labeled_dataset)
        self.train_labeled_dataloader = DataLoader(
            self.train_labeled_dataset,
            sampler=train_labeled_sampler,
            batch_size=args.eval_batch_size
        )


        self.train_unlabeled_dataset = new_data.train_unlabeled_dataset
        self.train_unlabeled_examples = new_data.train_unlabeled_examples
        self.train_unlabeled_ex_list = new_data.train_unlabeled_ex_list
        train_unlabeled_sampler = SequentialSampler(self.train_unlabeled_dataset)
        self.train_unlabeled_dataloader = DataLoader(
            self.train_unlabeled_dataset,
            sampler=train_unlabeled_sampler,
            batch_size=args.eval_batch_size
        )


        self.test_dataset = new_data.test_dataset
        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=args.eval_batch_size
        )

    def al_finetune(self, args):

        self.logger.info('Start active learning finetune ...')
        self.llm_labeling(args, epoch=0, model=self.model)


        best_model = copy.deepcopy(self.model)
        best_metrics = {
            'Epoch': 0,
            'ACC': 0,
            'ARI': 0,
            'NMI': 0
        }
        self.logger.info("Start train ...")
        indices = self.get_neighbor_inds(args, self.llm_augmented_dataset, self.llm_augmented_dataloader)
        self.get_neighbor_dataset(args, self.llm_augmented_dataset, indices)


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
            results = self.test(test_dataloader=self.test_dataloader, model=self.model, num_labels=self.num_labels)

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
                best_model = copy.deepcopy(self.model)
                # save_model(args, best_model, epoch)

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
                self.llm_labeling(args, epoch=epoch, model=best_model)
                indices = self.get_neighbor_inds(args, self.llm_augmented_dataset, self.llm_augmented_dataloader)
                self.get_neighbor_dataset(args, self.llm_augmented_dataset, indices)



    def alignment(self, old_centroids, new_centroids, cluster_labels):
        self.logger.info("***** Conducting Alignment *****")
        if old_centroids is not None:

            old_centroids = old_centroids
            new_centroids = new_centroids
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            aligned_centroids = np.zeros_like(old_centroids)
            alignment_labels = list(col_ind)

            for i in range(self.num_labels):
                label = alignment_labels[i]
                aligned_centroids[i] = new_centroids[label]
            # 新label对应老label
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in cluster_labels])

        else:
            aligned_centroids = new_centroids    
            pseudo_labels = cluster_labels 

        self.logger.info("***** Update Pseudo Labels With Real Labels *****")
        
        return aligned_centroids, pseudo_labels
    
    
    def llm_labeling(self, args, epoch, model):

        if not os.path.exists(os.path.join(args.result_dir, f'llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json')):
            os.makedirs(os.path.dirname(os.path.join(args.result_dir, f'llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json')), exist_ok=True)
            
            self.logger.info('Start LLM labeling ...')
            # 先按照整体的分布聚类
            feats, y_true = self.get_outputs(dataloader=self.train_semi_dataloader, model=model)
            km = KMeans(n_clusters = self.num_labels).fit(feats)
            cluster_centroids, y_pred = km.cluster_centers_, km.labels_
            cluster_centroids, y_pred = self.alignment(self.centroids, cluster_centroids, y_pred)
            self.centroids = cluster_centroids

            #匈牙利算法将预测结果映射到真实标签：y_pred_map: 每个具体样本预测标签对应映射后的标签，cluster_map: 每个聚类中心对应映射后的标签
            y_pred_map, cluster_map, cluster_map_opp = self.get_hungray_aligment(y_pred, y_true)
            # 选取离每个centroid最近的一个样本，用于Query LLM做选择, 得到 [num_centroids, 1]的indices,这个indices是在train_semi_dataset中的indices
            # labs是已经映射过的labels
            nearest_centroid_ex_inds,  nearest_centroid_ex_labs = self.get_nearst_centroid_example(feats, cluster_centroids, cluster_map, num_examples=1)    
            
            # 挑选没有标签的样本
            selected_unlabeled_ex_indices = np.where(self.llm_labels == -1)[0]
            selected_feats = feats[selected_unlabeled_ex_indices]
            selected_y_true = y_true[selected_unlabeled_ex_indices]
            selected_y_pred = y_pred[selected_unlabeled_ex_indices]
            selected_y_pred_map = y_pred_map[selected_unlabeled_ex_indices]

            # uncertainties 形状为 (selected_num_samples, ) 每一个样本的entropy(也是uncertainty)
            # unc_neighbors 形状为 (selected_num_samples, num_neighbours) 每一个样本都有k个最近邻索引, 用于label propagation
            # probs 形状为 (selected_num_samples, num_clusters) 每一个样本的分配给不同cluster的概率
            # sim_scores 形状为 (selected_num_samples, num_neighbours) 每一个样本的k个最近邻的相似度
            uncertainties, unc_neighbors, probs, sim_scores = self.get_uncertainty(args, selected_feats, cluster_centroids)
            assert unc_neighbors.size == sim_scores.size

            # 用于给不确定性最大的样本选取最近的clusters
            faiss_clusters = faiss.IndexFlatL2(cluster_centroids.shape[1])
            faiss_clusters.add(cluster_centroids)

            llm_generated_outputs = {
                "utterance_ori_inds": [],
                "true_cluster_ids": [],
                "pred_cluster_ids": [],
                "llm_pred_cluster_ids": [],
                "unc_neighbors_ori_inds": [],
                "unc_neighbors_labs": [],
            }
            price_usage = 0
            for i, cluster_ind in tqdm(enumerate(np.unique(y_pred)), desc="LLM Labeling Iter", total=len(np.unique(y_pred)), leave=False, dynamic_ncols=True):

                cluster_ex_inds = np.where(selected_y_pred == int(cluster_ind))[0]
                cluster_ex_ori_inds = selected_unlabeled_ex_indices[cluster_ex_inds]
                cluster_ex_uncertainties = uncertainties[cluster_ex_inds]

                if len(cluster_ex_inds) == 0:
                    continue
                max_uncertain_ex_id = np.argmax(cluster_ex_uncertainties)
                ex_ind = cluster_ex_inds[max_uncertain_ex_id]
                ex_ori_ind = cluster_ex_ori_inds[max_uncertain_ex_id]

                assert np.all(selected_feats[ex_ind] == feats[ex_ori_ind])
                # 获取该样本的representation
                max_uncertain_ex_feat = selected_feats[ex_ind]
                # 获取该样本的uncertain_neighbors, sim_scores
                max_unex_unc_neighbors = unc_neighbors[ex_ind]
                max_unex_sim_scores = sim_scores[ex_ind]

                # 获取离该样本最近的clusters
                num_nearst_clusters = self.num_labels if args.dataset == 'stackoverflow' else self.num_labels // 2
                _, nearest_cluster_inds = faiss_clusters.search(max_uncertain_ex_feat.reshape(1, -1), num_nearst_clusters)

                nearest_cluster_inds = nearest_cluster_inds.flatten()
                # 获取这些clusters的representative examples
                cluster_representative_ex_inds = nearest_centroid_ex_inds[nearest_cluster_inds]
                cluster_representative_ex_labs = nearest_centroid_ex_labs[nearest_cluster_inds]

                # 开始构建query LLM的prompt
                utterance_set = ''
                for j, indice in enumerate(cluster_representative_ex_inds):
                    utterenace_text = self.train_semi_dataset[indice]['input_text']

                    # 确保这些样本的对应的，并没有用到label信息
                    assert y_true[indice] == self.train_semi_dataset[indice]['label_id_true']
                    utterance_set += f'Cluster_id: {cluster_representative_ex_labs[j]}. Utterance: {utterenace_text}\n'
                
                
                max_uncertain_utterance_text = self.train_semi_dataset[ex_ori_ind]['input_text']

                # print('**** utterance_set: ', utterance_set)
                # print('**** max_uncertain_utterance_text: ', max_uncertain_utterance_text)
                    
                
                if args.dataset == 'banking':
                    comparison_prompting = PROMPT_BANKING.format(utterance_set, max_uncertain_utterance_text)
                elif args.dataset == 'clinc':
                    comparison_prompting = PROMPT_CLINC.format(utterance_set, max_uncertain_utterance_text)
                elif args.dataset == 'stackoverflow':
                    comparison_prompting = PROMPT_STACKOVERFLOW.format(utterance_set, max_uncertain_utterance_text)
                else:
                    raise NotImplementedError('Dataset not implemented')                
                
                true_cluster_id = y_true[ex_ori_ind]
                pred_cluster_id = y_pred_map[ex_ori_ind]

                

                messages = [
                    {"role": "system", "content": 'You are a linguistic expert specializing in new intent discovery.'},
                    {"role": "user", "content": comparison_prompting},
                ]


                llm_response = chat_completion_with_backoff(
                    model='gpt-3.5-turbo',
                    messages=messages,
                    temperature=0.0, 
                    max_tokens=256)
                
                num_input_tokens = llm_response.usage['prompt_tokens']
                num_output_tokens = llm_response.usage['completion_tokens']
                actual_model = llm_response.model
                curr_usage = num_input_tokens * OPENAI_PIRCE[actual_model]['input_token'] + num_output_tokens * OPENAI_PIRCE[actual_model]['output_token']
                price_usage += curr_usage
                self.logger.info('**** curr_usage: %s, total_usage: %s', curr_usage, price_usage)

                llm_response = llm_response.choices[0]['message']['content']

                # convert to number
                llm_pred_cluster_id = re.findall(r'-?\d+', llm_response)
                if len(llm_pred_cluster_id) == 0 or int(llm_pred_cluster_id[0]) >= len(cluster_map):
                    llm_pred_cluster_id = -1
                else:
                    llm_pred_cluster_id = int(llm_pred_cluster_id[0])

                # label propagation
                neighbor_labels = []
                neighbor_ori_inds = []
                for neighbor_ind, neighbor_sim_score in zip(max_unex_unc_neighbors, max_unex_sim_scores):
                    if llm_pred_cluster_id == -1:
                        neighbor_labels.append(-1)
                        neighbor_ori_inds.append(selected_unlabeled_ex_indices[neighbor_ind])
                    else:
                        # 原始未经过map的cluster顺序
                        neighbor_prob = probs[neighbor_ind]
                        # 将llm_pred_cluster_id反映射到原始的cluster顺序
                        ori_llm_pred_cluster_id = cluster_map_opp[llm_pred_cluster_id]
                        # check for correctness
                        assert cluster_map[ori_llm_pred_cluster_id] == llm_pred_cluster_id
                        llm_refine_prob = np.zeros_like(neighbor_prob)
                        llm_refine_prob[ori_llm_pred_cluster_id] = 1.0
                        refine_neighbor_prob = neighbor_prob + neighbor_sim_score * llm_refine_prob
                        # 选取最大的概率。因为是原始顺序，因此需要map到新顺序
                        refine_neighbor_lab = cluster_map[np.argmax(refine_neighbor_prob)]
                        if refine_neighbor_lab == llm_pred_cluster_id:
                            neighbor_labels.append(cluster_map[np.argmax(refine_neighbor_prob)])
                        else:
                            neighbor_labels.append(-1)
                        neighbor_ori_inds.append(selected_unlabeled_ex_indices[neighbor_ind])
                neighbor_labels = np.asarray(neighbor_labels)
                neighbor_ori_inds = np.asarray(neighbor_ori_inds)

                self.logger.info('**** llmg_cluster_id: %s', llm_pred_cluster_id)
                self.logger.info('**** true_cluster_id: %s', true_cluster_id)
                self.logger.info('**** pred_cluster_id: %s', pred_cluster_id)
                self.logger.info('**** cluster_representative_ex_labs: %s', cluster_representative_ex_labs)
                
                llm_generated_outputs["utterance_ori_inds"].append(int(ex_ori_ind))
                llm_generated_outputs["true_cluster_ids"].append(int(true_cluster_id))
                llm_generated_outputs["pred_cluster_ids"].append(int(pred_cluster_id))
                llm_generated_outputs["llm_pred_cluster_ids"].append(int(llm_pred_cluster_id))
                llm_generated_outputs["unc_neighbors_ori_inds"].append(neighbor_ori_inds.tolist())
                llm_generated_outputs["unc_neighbors_labs"].append(neighbor_labels.tolist())

                
                with open(os.path.join(args.result_dir, f'llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json'), "w") as fp:
                    json.dump(llm_generated_outputs, fp, indent=4)
        else:
            self.logger.info('Loading LLM annotated output from %s', os.path.join(args.result_dir, f'llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json'))
            with open(os.path.join(args.result_dir, f'llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json'), "r") as fp:
                llm_generated_outputs = json.load(fp)
        # 更新label
        self.logger.info('updating labels')
        self.updating_dataset(args, llm_generated_outputs)

    
    def updating_dataset(self, args, llm_generated_outputs):

        utterance_ori_inds = llm_generated_outputs["utterance_ori_inds"]
        true_cluster_ids = llm_generated_outputs["true_cluster_ids"]
        pred_cluster_ids = llm_generated_outputs["pred_cluster_ids"]
        llm_pred_cluster_ids = llm_generated_outputs["llm_pred_cluster_ids"]
        unc_neighbors_ori_inds = llm_generated_outputs["unc_neighbors_ori_inds"]
        unc_neighbors_labs = llm_generated_outputs["unc_neighbors_labs"]

        self.logger.info('Num of labeled examples before LLM predicted: %s', str(len(np.where(self.llm_labels != -1)[0])))
        for i, u_ori_ind in enumerate(utterance_ori_inds):
            # double check for correctness
            assert self.train_semi_dataset[u_ori_ind]['label_id_true'] == true_cluster_ids[i]
            # update self.llm_labels
            llm_pred_cluster_id = llm_pred_cluster_ids[i]
            self.llm_labels[u_ori_ind] = llm_pred_cluster_id
            # assign labels to neighbors
            for j, neighbor_ori_ind in enumerate(unc_neighbors_ori_inds[i]):
                self.llm_labels[neighbor_ori_ind] = unc_neighbors_labs[i][j]
        self.logger.info('Num of labeled examples after LLM predicted: %s', str(len(np.where(self.llm_labels != -1)[0])))

        # update dataset
        llm_augmented_list = []
        for ind, llm_label in enumerate(self.llm_labels):
            self.train_semi_dataset[ind]['label_id'] = llm_label

            if llm_label != -1:
                llm_augmented_list.append(self.train_semi_dataset[ind])

        train_semi_sampler = SequentialSampler(self.train_semi_dataset)
        self.train_semi_dataloader = DataLoader(
            self.train_semi_dataset,
            sampler=train_semi_sampler,
            batch_size=args.train_batch_size
        )

        self.llm_augmented_dataset = NIDDataset(llm_augmented_list)
        llm_augmented_sampler = SequentialSampler(self.llm_augmented_dataset)
        self.llm_augmented_dataloader = DataLoader(
            self.llm_augmented_dataset,
            sampler=llm_augmented_sampler,
            batch_size=args.train_batch_size
        )

    
    def get_nearst_centroid_example(self, feats, cluster_centroids, cluster_map, num_examples=1):
        self.logger.info('Get utterances nearst to centroids')
        assert feats.shape[1] == cluster_centroids.shape[1]
        index = faiss.IndexFlatL2(feats.shape[1])
        index.add(feats)
        D, I = index.search(cluster_centroids, num_examples)
        I = I.flatten() # (num_centroids, num_examples) -> (num_centroids * num_examples)
        ex_labels = []
        for i, indice in enumerate(I):
            if indice < len(self.labeled_ex_labels):
                ex_labels.append(self.labeled_ex_labels[indice])
            else:
                ex_labels.append(cluster_map[i])
        return I, np.asarray(ex_labels)


    def get_uncertainty(self, args, feats, cluster_centroids):
        '''https://github.com/THU-BPM/SelfORE/blob/master/adaptive_clustering.py'''
        self.logger.info('Calculating student t distribution')
        assert feats.shape[1] == cluster_centroids.shape[1]
        # get student t distribution
        distances = euclidean_distances(feats, cluster_centroids)**2 # euclidean_distances会开根号
        # 计算学生t分布(已确认没问题)
        st_distribution = (1.0 + distances / args.student_t_freedom) ** (- (args.student_t_freedom + 1) / 2)
        st_distribution = st_distribution / np.sum(st_distribution, axis=1, keepdims=True)
        st_distribution = np.clip(st_distribution, 1e-12, 1.0) # 防止越界产生nan
        # get entropies
        entropies = - np.sum(st_distribution * np.log(st_distribution), axis=1)
        # refine uncertainty with neighbors
        self.logger.info('Refining uncertainty with neighbors')
        index = faiss.IndexFlatL2(feats.shape[1])
        index.add(feats)
        # 先确定5个最近邻
        D, I = index.search(feats, args.uncertainty_neighbour_num + 1)
        D, I = D[:, 1:], I[:, 1:] # 去掉自己
        similarity_scores = np.exp(-D*args.rho)
        weighted_similarity_scores = np.mean(entropies[I] * similarity_scores, axis=-1)
        entropies = entropies + weighted_similarity_scores
        return entropies, I, st_distribution, similarity_scores


    def get_hungray_aligment(self, y_pred, y_true):
        num_test_samples = len(y_pred)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D))
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
        y_pred_map = []
        cluster_map = [0]*len(ind)
        cluster_map_opp = [0]*len(ind)
        for i in range(num_test_samples):
            yp = y_pred[i]
            y_pred_map.append(ind[yp][1])
        y_pred_map = np.asarray(y_pred_map)

        for item in ind:
            cluster_map[item[0]] = item[1]
            cluster_map_opp[item[1]] = item[0]
        cluster_map = np.asarray(cluster_map)
        cluster_map_opp = np.asarray(cluster_map_opp)
        assert np.all(cluster_map[cluster_map_opp] == np.arange(len(ind)))
        return y_pred_map, cluster_map, cluster_map_opp

    def test(self, test_dataloader, model, num_labels):
        
        feats, y_true = self.get_outputs(dataloader=test_dataloader, model=model)
        km = KMeans(n_clusters=num_labels).fit(feats)
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
    
    def get_outputs(self, dataloader, model):

        model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0, model.config.hidden_size)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration", leave=False):
            batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.set_grad_enabled(False):
                sent_embed = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], 
                    labels=None, mode='feature_ext')
                total_labels = torch.cat((total_labels, batch['label_id_true']))
                total_features = torch.cat((total_features, sent_embed))
                
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        return feats, y_true
