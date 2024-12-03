import os
import sys

import logging
import argparse
import sys
import os
import datetime
import pickle as pkl

from utils import set_seed, load_yaml_config
from easydict import EasyDict
from dataloaders.base_data import BaseDataNew




def set_logger(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"{args.method}_{args.dataset}_{args.known_cls_ratio}_{time}_{args.labeled_ratio}.log"
    
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(args.log_dir, file_name))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='open_intent_discovery', 
                        help="Type for methods")

    parser.add_argument("--dataset", default='banking', type=str, 
                        help="The name of the dataset to train selected")
    
    parser.add_argument("--dataset_dir", default='', type=str,
                        help="The saving path of the dataset")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, 
                        help="The number of known classes")
    
    parser.add_argument("--labeled_ratio", default=0.1, type=float, 
                        help="The ratio of labeled samples in the training set")
    
    parser.add_argument("--cluster_num_factor", default=1.0, type=float, 
                        help="The factor (magnification) of the number of clusters K.")
    
    parser.add_argument("--cluster_k", default=1, type=int, 
                        help="The factor (magnification) of the number of clusters K.")
    
    parser.add_argument("--method", type=str, default='alnid', 
                        help="which method to use")

    parser.add_argument('--seed', type=int, default=0, 
                        help="random seed for initialization")
    
    parser.add_argument("--config_file_name", type=str, default = 'Prompt.py',
                        help="The config file name for the model.")

    parser.add_argument('--log_dir', type=str, default='logs', 
                        help="Logger directory.")
    
    parser.add_argument("--output_dir", default= './outputs', type=str, 
                        help="The output directory where all train data will be written.") 

    parser.add_argument("--result_dir", type=str, required=True,
                        default = 'results', help="The path to save results")

    parser.add_argument("--results_file_name", type=str, default = 'results.csv', 
                        help="The file name of all the results.")

    parser.add_argument("--model_file_name", type=str, default = '', 
                        help="The file name of trained model.")
    
    parser.add_argument("--pretrained_nidmodel_file_name", type=str, default = '', 
                        help="The file name of pretrained_nidmodel_file_name.")

    parser.add_argument("--save_results", action="store_true", 
                        help="save final results for open intent detection")
    
    parser.add_argument("--cl_loss_weight", default=1.0, type=float, 
                        help="loss_weight")

    parser.add_argument("--semi_cl_loss_weight", default=1.0, type=float, 
                        help="loss_weight")

    args = parser.parse_args()

    return args


def run():

    command_args = parse_arguments()
    configs = load_yaml_config(command_args.config_file_name)
    args = EasyDict(dict(vars(command_args), **configs))
    logger = set_logger(args)

    logger.debug("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.debug(f"{k}:\t{args[k]}")
    logger.debug("="*30+" End Params "+"="*30)

    set_seed(args.seed)

    logger.info('Data and Model Preparation...')

    dataset_path = os.path.join(args.output_dir, args.dataset_dir)
    try:
        logger.info('Loading the processed data...')
        with open(dataset_path, 'rb') as f:
            data = pkl.load(f)
    except (FileNotFoundError, EOFError, pkl.UnpicklingError) as e:
        # 捕获文件不存在或数据损坏的异常
        logger.info('Dataset not found or corrupted. Re-processing the training data.')
        
        dataset_dir = '/'.join(dataset_path.split('/')[:-1])
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        data = BaseDataNew(args)
        with open(dataset_path, 'wb') as f:
            pkl.dump(data, f)

    if args.mode == 'train':
        from methods.alup.pretrain_manager import PretrainManager
        from methods.alup.manager import Manager

        logger.info('Pretrain Begin...')
        pretrain_manager = PretrainManager(args, data)
        pretrain_manager.train(args)
        if args.finish_pretrain is not None and args.finish_pretrain:
            return
        manager = Manager(args, data, pretrained_model=pretrain_manager.model)
        manager.train(args, data)
        logger.info('Pretrain Finished...')
    elif args.mode == 'al_finetune':
        from methods.alup.al_manager import ALManager
        print('AL Fine-tuning Begin...')
        finetune_manager = ALManager(args, data, os.path.join(args.output_dir, f'models_{args.known_cls_ratio}', args.pretrained_nidmodel_file_name))
        finetune_manager.al_finetune(args)

    


    



if __name__ == '__main__':

    run()