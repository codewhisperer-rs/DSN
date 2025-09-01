import os
import pandas as pd
from log import *
from eval import *
from utils import *
from utils import get_args, EarlyStopMonitor, FixedNegSampler, set_random_seed
from train import *
from module import POLAR
from graph import NeighborFinder
import random

import time
import datetime
import wandb
now = datetime.datetime.now()
args, sys_argv = get_args()



LEARNING_RATE = args.lr 
NUM_NEIGHBORS = args.n_degree
NUM_LAYER = args.n_layer
DATA = args.data 

BATCH_SIZE =args.bs
EARLY_STOP = args.early_stop 
SEED = args.seed 
NUM_EPOCH = args.n_epoch 
GPU = args.gpu
TOLERANCE = args.tolerance 
POS_DIM = args.pos_dim  
TIME_DIM = args.time_dim 
TASK = args.task 
TIME_ENCODER_TYPE = args.train_time_encoding 
EDGE_EMBEDDING = args.edge_embedding
NEIGH_AGG = args.neigh_agg
PATH_AGG = args.path_agg
WALK_TYPE = args.walk_type
DIREC = args.direct
COOCC = args.co_occ

set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv, now)

Dynamic_path = os.path.dirname(os.path.dirname(__file__))

# Always use pre-split CSVs; raise if missing.
# Resolve splits directory
if args.splits_dir is None:
    splits_dir = os.path.join(Dynamic_path, '../DynamicData', 'splits', f'{DATA}_seed{SEED}')
else:
    splits_dir = args.splits_dir

# Load split CSVs
def _load_split(name):
    p = os.path.join(splits_dir, f'{name}.csv')
    if not os.path.exists(p):
        raise FileNotFoundError(f'Missing split file: {p}. Generate via export_splits.py or provide --splits_dir.')
    return pd.read_csv(p)

train_df = _load_split('train')
val_df = _load_split('val')
test_df = _load_split('test')
trans_df = _load_split('transductive')
induc_df = _load_split('inductive')

# Full dataset is concat of train/val/test
g_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Assign arrays
# Full arrays for building adjacency
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
sign_l = g_df.label.values
ts_l = g_df.ts.values
weight_l = g_df.weight.values

# Split arrays
def _arrs(df):
    return df.u.values, df.i.values, df.ts.values, df.idx.values, df.label.values, df.weight.values

train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l, train_weight_l = _arrs(train_df)
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l, val_weight_l = _arrs(val_df)
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l, test_weight_l = _arrs(test_df)
tr_test_src_l, tr_test_dst_l, tr_test_ts_l, tr_test_e_idx_l, tr_test_label_l, tr_test_weight_l = _arrs(trans_df)
nn_test_src_l, nn_test_dst_l, nn_test_ts_l, nn_test_e_idx_l, nn_test_label_l, nn_test_weight_l = _arrs(induc_df)

if DIREC == 'add':
    EDGE_FEAT_DIM = np.shape(weight_l.reshape(len(weight_l), 1))[1] + 2
else:
    EDGE_FEAT_DIM = np.shape(weight_l.reshape(len(weight_l), 1))[1]

max_idx = max(src_l.max(), dst_l.max())
total_node_set = set(np.unique(np.hstack([src_l, dst_l])))
num_total_unique_nodes = len(total_node_set)

train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l, train_weight_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l, val_weight_l
train_val_data = (train_data, val_data)

logger.info('Using pre-split CSVs from: {}'.format(splits_dir))
logger.info('Train set: {}, Val set: {}, Test set: {}, Trans set: {}, Induc set: {}'.format(len(train_src_l), len(val_src_l), len(test_src_l), len(tr_test_src_l), len(nn_test_src_l)))
logger.info('-------------------')

if DIREC == 'non':
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts, sign, weight in zip(src_l, dst_l, e_idx_l, ts_l, sign_l, weight_l):
        full_adj_list[src].append((dst, eidx, ts, sign, weight))
        full_adj_list[dst].append((src, eidx, ts, sign, weight))
    
    full_ngh_finder = NeighborFinder(full_adj_list, walk_type=WALK_TYPE, bias=args.bias, sample_method=args.pos_sample)

    partial_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts, sign, weight in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l, train_label_l, train_weight_l):
        partial_adj_list[src].append((dst, eidx, ts, sign, weight))
        partial_adj_list[dst].append((src, eidx, ts, sign, weight))

    for src, dst, eidx, ts, sign, weight in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l, val_label_l, val_weight_l):
        partial_adj_list[src].append((dst, eidx, ts, sign, weight))
        partial_adj_list[dst].append((src, eidx, ts, sign, weight))
        

   
    partial_ngh_finder = NeighborFinder(partial_adj_list, walk_type=WALK_TYPE,bias=args.bias, sample_method=args.pos_sample)

elif DIREC == 'add':
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts, sign, weight in zip(src_l, dst_l, e_idx_l, ts_l, sign_l, weight_l):

        full_adj_list[src].append((dst, eidx, ts, sign, weight, 1))
        full_adj_list[dst].append((src, eidx, ts, sign, weight, -1))


    full_ngh_finder = NeighborFinder(full_adj_list, walk_type=WALK_TYPE, bias=args.bias, sample_method=args.pos_sample)

    partial_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts, sign, weight in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l, train_label_l, train_weight_l):
        partial_adj_list[src].append((dst, eidx, ts, sign, weight, 1))
        partial_adj_list[dst].append((src, eidx, ts, sign, weight, -1))

    for src, dst, eidx, ts, sign, weight in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l, val_label_l, val_weight_l):
        partial_adj_list[src].append((dst, eidx, ts, sign, weight, 1))
        partial_adj_list[dst].append((src, eidx, ts, sign, weight, -1))
        

    partial_ngh_finder = NeighborFinder(partial_adj_list, walk_type=WALK_TYPE,bias=args.bias, sample_method=args.pos_sample)

ngh_finders = partial_ngh_finder, full_ngh_finder

# Negative samplers (require fixed CSVs; no dynamic sampling)
splits_dir = args.splits_dir or os.path.join(Dynamic_path, '../DynamicData', 'splits', f'{DATA}_seed{SEED}')

def _load_fixed_neg(name: str) -> FixedNegSampler:
    p = os.path.join(splits_dir, f'{name}_neg.csv')
    if not os.path.exists(p):
        raise FileNotFoundError(
            'Missing negative samples CSV: {}. Generate via export_splits.py with --export_neg, '
            'or provide --splits_dir pointing to prepared files.'.format(p)
        )
    df = pd.read_csv(p)
    logger.info('Using fixed {} negatives from {}'.format(name, p))
    return FixedNegSampler(df.u.values, df.i.values, seed=SEED)

train_rand_sampler = _load_fixed_neg('train')
val_rand_sampler = _load_fixed_neg('val')
test_rand_sampler = _load_fixed_neg('test')
trans_test_rand_sampler = _load_fixed_neg('transductive')
induc_test_rand_sampler = _load_fixed_neg('inductive')

rand_samplers = train_rand_sampler, val_rand_sampler

device = torch.device('cuda:{}'.format(GPU))
# device = torch.device('cpu')
polar = POLAR(task = TASK, num_layers=NUM_LAYER, 
            time_dim = TIME_DIM, pos_dim=POS_DIM, edge_feature_dim = EDGE_FEAT_DIM,
            num_neighbors=NUM_NEIGHBORS, 
            get_checkpoint_path=get_checkpoint_path, time_encoder_type=TIME_ENCODER_TYPE, edge_embedding_type=EDGE_EMBEDDING, neigh_agg=NEIGH_AGG, path_agg=PATH_AGG, co_occurence = COOCC, node_num = num_total_unique_nodes)

polar.to(device)
optimizer = torch.optim.Adam(polar.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
criterion_muli = torch.nn.CrossEntropyLoss()
early_stopper = EarlyStopMonitor(max_round=EARLY_STOP, tolerance=TOLERANCE) 

if TASK == 'link':   
    print("Task: link") 
    train_val(train_val_data, polar, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger)

elif TASK == 'link_sign': 
    print("Task: link and sign")  

    train_val_for_multiclass(partial_adj_list, train_val_data, polar, BATCH_SIZE, NUM_EPOCH, criterion_muli, optimizer, early_stopper, ngh_finders, rand_samplers, logger)

# final testing
polar.update_ngh_finder(full_ngh_finder)

if TASK == 'link':   
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test nodes', polar, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
    logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}, f1: {}'.format('test', test_acc, test_auc, test_ap, test_f1))
    
    transductive_acc, transductive_ap, transductive_auc, transductive_f1 = [-1]*4
    transductive_acc, transductive_ap, transductive_f1, transductive_auc = eval_one_epoch('test for transductive nodes', polar, trans_test_rand_sampler, tr_test_src_l, tr_test_dst_l, tr_test_ts_l, tr_test_label_l, tr_test_e_idx_l)
    logger.info('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}, f1: {}'.format('trans', transductive_acc, transductive_ap, transductive_auc, transductive_f1))

    inductive_acc, inductive_ap, inductive_auc, inductive_f1 = [-1]*4
    inductive_acc, inductive_ap, inductive_f1, inductive_auc = eval_one_epoch('test for inductive nodes', polar, induc_test_rand_sampler, nn_test_src_l, nn_test_dst_l, nn_test_ts_l, nn_test_label_l, nn_test_e_idx_l)
    logger.info('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}, f1: {}'.format('induc', inductive_acc, inductive_auc, inductive_ap, inductive_f1))
    

elif TASK == 'link_sign':
    test_precision, test_accuracy, test_recall, test_weighted_f1, test_micro_f1, test_macro_f1 = eval_one_epoch_for_multiclass(full_adj_list, 'test nodes', polar, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_weight_l, test_e_idx_l)
    logger.info('Test statistics: {} all nodes -- '.format('test'))
    logger.info('{}\t{}\t{}\t{}\t{}\t{}'.format(test_precision, test_accuracy, test_recall, test_weighted_f1, test_micro_f1, test_macro_f1))

    transductive_precision, transductive_accuracy, transductive_recall, transductive_weighted_f1, transductive_micro_f1, transductive_macro_f1 = [-1]*6
    transductive_precision, transductive_accuracy, transductive_recall, transductive_weighted_f1, transductive_micro_f1, transductive_macro_f1 = eval_one_epoch_for_multiclass(full_adj_list, 'test for inductive nodes', polar, trans_test_rand_sampler, tr_test_src_l, tr_test_dst_l, tr_test_ts_l, tr_test_label_l, tr_test_weight_l, tr_test_e_idx_l)
    logger.info('Test statistics: {} new-new nodes -- '.format('trans'))
    logger.info('{}\t{}\t{}\t{}\t{}\t{}'.format(transductive_precision, transductive_accuracy, transductive_recall, transductive_weighted_f1, transductive_micro_f1, transductive_macro_f1))
    

    inductive_precision, inductive_accuracy, inductive_recall, inductive_weighted_f1, inductive_micro_f1, inductive_macro_f1 = [-1]*6
    inductive_precision, inductive_accuracy, inductive_recall, inductive_weighted_f1, inductive_micro_f1, inductive_macro_f1 = eval_one_epoch_for_multiclass(full_adj_list, 'test for inductive nodes', polar, induc_test_rand_sampler, nn_test_src_l, nn_test_dst_l, nn_test_ts_l, nn_test_label_l, nn_test_weight_l, nn_test_e_idx_l)
    logger.info('Test statistics: {} new-new nodes -- '.format('induc'))
    logger.info('{}\t{}\t{}\t{}\t{}\t{}'.format(inductive_precision, inductive_accuracy, inductive_recall, inductive_weighted_f1, inductive_micro_f1, inductive_macro_f1))
    


# save model
logger.info('Saving POLAR model ...')
torch.save(polar.state_dict(), best_model_path)
logger.info('POLAR model saved')

# save one line result
if TASK == 'link':
    save_oneline_result('log/', args, [test_acc, test_ap, test_f1, test_auc, inductive_acc, inductive_ap, inductive_auc], now)
elif TASK == 'link_sign':   
    save_oneline_result('log/', args, [test_precision,  test_accuracy, test_recall, test_weighted_f1, test_micro_f1, test_macro_f1, inductive_precision, inductive_accuracy, inductive_recall, inductive_weighted_f1, inductive_micro_f1, inductive_macro_f1], now)
