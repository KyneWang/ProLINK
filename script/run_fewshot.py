import os
import sys
import csv
import math
import json
import time
import pprint
import argparse
import random
import datetime
import itertools
from tqdm import tqdm

import numpy as np
import torch
from torch import distributed as dist
from torch.utils import data as torch_data
from torch.nn.functional import cosine_similarity
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra, Ultra2
from script.run import train_and_validate, test

os.environ["CUDA_LAUNCH_BLOCKING"] = '1' # CUDA calls will become synchronous execution.

default_finetuning_config = {
    # graph: (num_epochs, batches_per_epoch), null means all triples in train set
    # transductive datasets (17)
    # standard ones (10)
    "CoDExSmall": (1, 4000),
    "CoDExMedium": (1, 4000),
    "CoDExLarge": (1, 2000),
    "FB15k237": (1, 'null'),
    "WN18RR": (1, 'null'),
    "YAGO310": (1, 2000),
    "DBpedia100k": (1, 1000),
    "AristoV4": (1, 2000),
    "ConceptNet100k": (1, 2000),
    "ATOMIC": (1, 200),
    # tail-only datasets (2)
    "NELL995": (1, 'null'),  # not implemented yet
    "Hetionet": (1, 4000),
    # sparse datasets (5)
    "WDsinger": (3, 'null'),
    "FB15k237_10": (1, 'null'),
    "FB15k237_20": (1, 'null'),
    "FB15k237_50": (1, 1000),
    "NELL23k": (3, 'null'),
    # inductive datasets (42)
    # GraIL datasets (12)
    "FB15k237Inductive": (1, 'null'),    # for all 4 datasets
    "WN18RRInductive": (1, 'null'),      # for all 4 datasets
    "NELLInductive": (3, 'null'),        # for all 4 datasets
    # ILPC (2)
    "ILPC2022SmallInductive": (3, 'null'),
    "ILPC2022LargeInductive": (1, 1000),
    # Ingram datasets (13)
    "NLIngram": (3, 'null'),  # for all 5 datasets
    "FBIngram": (3, 'null'),  # for all 4 datasets
    "WKIngram": (3, 'null'),  # for all 4 datasets
    # MTDEA datasets (10)
    "WikiTopicsMT1": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT2": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT3": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT4": (3, 'null'),  # for all 2 test datasets
    "Metafam": (3, 'null'),
    "FBNELL": (3, 'null'),
    # Hamaguchi datasets (4)
    "HM": (1, 100)  # for all 4 datasets
}

default_train_config = {
    # graph: (num_epochs, batches_per_epoch), null means all triples in train set
    # transductive datasets (17)
    # standard ones (10)
    "CoDExSmall": (10, 1000),
    "CoDExMedium": (10, 1000),
    "CoDExLarge": (10, 1000),
    "FB15k237": (10, 1000),
    "WN18RR": (10, 1000),
    "YAGO310": (10, 2000),
    "DBpedia100k": (10, 1000),
    "AristoV4": (10, 1000),
    "ConceptNet100k": (10, 1000),
    "ATOMIC": (10, 1000),
    # tail-only datasets (2)
    "NELL995": (10, 1000),  # not implemented yet
    "Hetionet": (10, 1000),
    # sparse datasets (5)
    "WDsinger": (10, 1000),
    "FB15k237_10": (10, 1000),
    "FB15k237_20": (10, 1000),
    "FB15k237_50": (10, 1000),
    "NELL23k": (10, 1000),
    # inductive datasets (42)
    # GraIL datasets (12)
    "FB15k237Inductive": (10, 'null'),    # for all 4 datasets
    "WN18RRInductive": (10, 'null'),      # for all 4 datasets
    "NELLInductive": (10, 'null'),        # for all 4 datasets
    # ILPC (2)
    "ILPC2022SmallInductive": (10, 'null'),
    "ILPC2022LargeInductive": (10, 1000),
    # Ingram datasets (13)
    "NLIngram": (10, 'null'),  # for all 5 datasets
    "FBIngram": (10, 'null'),  # for all 4 datasets
    "WKIngram": (10, 'null'),  # for all 4 datasets
    # MTDEA datasets (10)
    "WikiTopicsMT1": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT2": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT3": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT4": (10, 'null'),  # for all 2 test datasets
    "Metafam": (10, 'null'),
    "FBNELL": (10, 'null'),
    # Hamaguchi datasets (4)
    "HM": (10, 1000)  # for all 4 datasets
}

default_reltype_dict={
    "FBIngram:25" : "FB15k237",
    "FBIngram:50" : "FB15k237",
    "FBIngram:75" : "FB15k237",
    "FBIngram:100" : "FB15k237",
    "WKIngram:25" : "Wikidata",
    "WKIngram:50" : "Wikidata",
    "WKIngram:75" : "Wikidata",
    "WKIngram:100" : "Wikidata",
    "NLIngram:0" : "NELL995",
    "NLIngram:25" : "NELL995",
    "NLIngram:50" : "NELL995",
    "NLIngram:75" : "NELL995",
    "NLIngram:100" : "NELL995",
    "WikiTopicsMT1:tax" : "Wikidata",
    "WikiTopicsMT1:health" : "Wikidata",
    "WikiTopicsMT2:org" : "Wikidata",
    "WikiTopicsMT2:sci" : "Wikidata",
    "WikiTopicsMT3:art" : "Wikidata",
    "WikiTopicsMT3:infra" : "Wikidata",
    "WikiTopicsMT4:sci" : "Wikidata",
    "WikiTopicsMT4:health" : "Wikidata",
}

candidate_reltype_list = ["genre/type", "person", "animal", "location/place", "organization", "creative work", "time", "profession", "event", "actual item", "language"]

separator = ">" * 30
line = "-" * 30

def set_seed(seed):
    random.seed(seed + util.get_rank())
    # np.random.seed(seed + util.get_rank())
    torch.manual_seed(seed + util.get_rank())
    torch.cuda.manual_seed(seed + util.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# the rel ids of batch samples are the same
class GroupedDataLoader(torch_data.DataLoader):
    def __init__(self, triples, graph_data, batch_size=1, shuffle=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.triples = triples
        self.shuffle = shuffle
        indices = {}
        
        for i, sample in enumerate(triples):
            key = sample[2].item() # h,t,r
            if key not in indices:
                indices[key] = []
            indices[key].append(i)
        self.grouped_indices = indices
        # remove the sample group whose relation is the backbone of the KG

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.grouped_indices)
        for rel, indices in self.grouped_indices.items():
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                yield self.collate_fn([self.triples[i] for i in batch_indices])

def get_unique_connections(edge_set, n_rel, ori_rel=None):
    if isinstance(edge_set[0], str):
        edge_set = torch.tensor([[int(item) for item in edge_key.split("_")] for edge_key in edge_set], dtype=torch.long).cuda()
    # adjust h 和 t to [0, n_rel-1]
    h_adjusted = torch.where(edge_set[:, 0] < n_rel, edge_set[:, 0], edge_set[:, 0] - n_rel)
    t_adjusted = torch.where(edge_set[:, 1] < n_rel, edge_set[:, 1], edge_set[:, 1] - n_rel)
    # condition mask
    condition_mask = (h_adjusted == edge_set[:, 0]) & (t_adjusted == edge_set[:, 1])
    if ori_rel is not None:
        ori_rel_mask = (edge_set[:, 0] == ori_rel) | ((edge_set[:, 0] != ori_rel) & (edge_set[:, 1] != ori_rel))
        condition_mask &= ori_rel_mask
    filtered_edges = edge_set[condition_mask]
    # filtering
    if ori_rel is None:
        unique_mask = filtered_edges[:, 0] <= filtered_edges[:, 1]
    else:
        unique_mask = filtered_edges[:, 0] == ori_rel
    unique_edges = filtered_edges[unique_mask]
    return unique_edges

def get_rels_fromConnectType(edge_tensor, target_rel, target_type):
    edge_mask = edge_tensor[:,2] == target_type
    edges = edge_tensor[edge_mask]
    rel_mask = (edges[:,0] == target_rel) | (edges[:,1] == target_rel)
    rel_edges = edges[rel_mask]
    # extract entities
    output_rels = torch.cat((rel_edges[:,0], rel_edges[:,1])).unique()
    output_rels = output_rels[output_rels != target_rel]
    if len(output_rels) == 0:
        output_rels = torch.Tensor([]).to(edge_tensor.device)
    return output_rels
    
def extend_rel_connections(ori_rel, unique_rel_edges, unique_oth_edges, connect_type=0):
    new_connection_list = [torch.Tensor([[ori_rel, ori_rel, 0]]).long().to(unique_rel_edges.device)]
    for connect_type in range(4):
        type2 = [0, 1, 1, 0][connect_type]
        head_rels = get_rels_fromConnectType(unique_rel_edges, ori_rel, connect_type)
        if len(head_rels) > 0:
            mask = (unique_oth_edges[:, 0].unsqueeze(1) == head_rels) | \
                   (unique_oth_edges[:, 1].unsqueeze(1) == head_rels)
            mask = mask.any(dim=1) & (unique_oth_edges[:, 2] == type2) 
            hop2_edges = unique_oth_edges[mask]
            hop2_rels = torch.cat((hop2_edges[:, 0], hop2_edges[:, 1])).unique()
            hop2_rels = hop2_rels[(hop2_rels.unsqueeze(1) != head_rels).any(dim=1)]
            if len(hop2_rels) > 0:
                new_connections = torch.stack([torch.full_like(hop2_rels, ori_rel), hop2_rels,
                                           torch.full_like(hop2_rels, connect_type)], dim=1).unique(dim=0)
                new_connection_list.append(new_connections.long())
    if len(new_connection_list) > 1:
        new_connections = torch.cat(new_connection_list, dim=0).unique(dim=0)
    else:
        new_connections = new_connection_list[0]
    return new_connections

def build_inverse_connections(new_edges, n_rel):
    device = new_edges.device
    new_edges = new_edges.detach().cpu().numpy().tolist()
    relgraph_mat = np.zeros((n_rel * 2, n_rel * 2, 4))
    for hid,tid,rid in new_edges:
        if rid in [0, 1]: 
            if rid == 0: values = [0,1,2,3]
            elif rid == 1: values = [1,0,3,2]
            relgraph_mat[hid,tid,values[0]] = 1  # h2h when rid = 0
            relgraph_mat[tid,hid,values[0]] = 1
            relgraph_mat[hid+n_rel,tid+n_rel,values[1]] = 1  # t2t
            relgraph_mat[tid+n_rel,hid+n_rel,values[1]] = 1
            relgraph_mat[hid,tid+n_rel,values[2]] = 1 # h2t
            relgraph_mat[tid,hid+n_rel,values[2]] = 1
            relgraph_mat[hid+n_rel,tid,values[3]] = 1 # t2h
            relgraph_mat[tid+n_rel,hid,values[3]] = 1
        elif rid in [2, 3]:
            if rid == 2: values = [2,3,0,1]
            elif rid == 3: values = [3,2,1,0]
            relgraph_mat[hid,tid,values[0]] = 1  # h2t when rid = 2
            relgraph_mat[tid+n_rel,hid+n_rel,values[0]] = 1
            relgraph_mat[tid,hid,values[1]] = 1 # t2h
            relgraph_mat[hid+n_rel,tid+n_rel,values[1]] = 1  
            relgraph_mat[hid,tid+n_rel,values[2]] = 1 # h2h
            relgraph_mat[tid+n_rel,hid,values[2]] = 1
            relgraph_mat[hid+n_rel,tid,values[3]] = 1 # t2t
            relgraph_mat[tid,hid+n_rel,values[3]] = 1 
    indexes = np.nonzero(relgraph_mat)
    all_new_edges = torch.LongTensor(np.stack([indexes[0],indexes[1],indexes[2]],axis=1)).to(device)
    return all_new_edges

def compute_remove_rels(unique_candidate_edges, unique_error_edges, ori_rel):
    remove_rel_dict = [[None, None] for _ in range(4)]
    head_error_edges = unique_error_edges[unique_error_edges[:, 2] == 0] 
    tail_error_edges = unique_error_edges[unique_error_edges[:, 2] == 1] 
        
    for connect_type in range(4):
        error_edges = head_error_edges if connect_type in [0, 2] else tail_error_edges 
        edge_side = "head" if connect_type in [0, 2] else "tail"
        cand_rels = get_rels_fromConnectType(unique_candidate_edges, ori_rel, connect_type)

        if len(cand_rels) > 0:
            cand_hit_counts = torch.zeros_like(cand_rels)
            cand_rel_ids = {item.item():rid for rid, item in enumerate(cand_rels)}
            
            mask1 = (error_edges[:, 0].unsqueeze(1) == cand_rels) 
            mask2 = (error_edges[:, 1].unsqueeze(1) == cand_rels)
            global_masked_error_edges = error_edges[mask1.any(dim=1) & mask2.any(dim=1)]
            
            for i, rel in enumerate(cand_rels):
                cand_hit_counts[i] = ((global_masked_error_edges[:, 0] == rel) | (global_masked_error_edges[:, 1] == rel)).sum() #.float().mean()
            remove_rel_dict[connect_type] = [cand_rels, cand_hit_counts]
    return remove_rel_dict

def remove_rels_from_candidates(unique_candidate_edges, remove_rel_dict, threshold=5):
    total_remove_rels = [set() for _ in range(4)]
    for connect_type in range(4):
        cand_rels, cand_hit_counts = remove_rel_dict[connect_type]
        if cand_rels is None: continue
        if threshold == -1:
            threshold = cand_hit_counts.float().mean().item()
        elif threshold == -2:
            threshold = cand_hit_counts.max().item() + 1
        remove_rels_mask = cand_hit_counts >= threshold
        total_remove_rels[connect_type] = set(cand_rels[remove_rels_mask].tolist())

    mask = torch.ones(len(unique_candidate_edges), dtype=torch.bool)
    for i, edge in enumerate(unique_candidate_edges):
        if edge[1].item() in total_remove_rels[edge[2]]:
            mask[i] = False
    filtered_candidate_edges = unique_candidate_edges[mask]
    return filtered_candidate_edges
    
@torch.no_grad()
def test_seq(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False, mode=None, fs_ds_dict=None, ds_name=None, k_num=0):
    world_size = util.get_world_size()
    rank = util.get_rank()
    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    test_loader = GroupedDataLoader(test_triplets, test_data, cfg.train.batch_size)
    n_rel = (test_data.num_relations // 2).item()
    model.eval()
    rankings = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
    
    rel_cache_connections = {}
    enxrg_cache_data = model.enxrg_cache_data # {}

    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        target_rel = batch[0, 2].item()
        if str(target_rel) not in fs_ds_dict: 
            continue 
        target_rel_inverse = target_rel + n_rel if target_rel < n_rel else target_rel - n_rel
        ori_rel = min(target_rel,target_rel_inverse)
        
        rel_mask1 = test_data.edge_type != target_rel
        rel_mask2 = test_data.edge_type != target_rel_inverse
        rel_mask = rel_mask1 & rel_mask2
        new_edge_index = test_data.edge_index.t()[rel_mask].t()
        new_edge_type = test_data.edge_type[rel_mask]
        
        zero_data = Data(
            edge_index= new_edge_index, 
            edge_type= new_edge_type,
            num_nodes=test_data.num_nodes, 
            num_relations=test_data.num_relations
        )
        zero_data = tasks.build_relation_graph(zero_data)
        
        if ds_name[:5]!="0shot":
            head_vec = torch.LongTensor(fs_ds_dict[str(target_rel)]["h"]).to(batch.device)
            tail_vec = torch.LongTensor(fs_ds_dict[str(target_rel)]["t"]).to(batch.device)

            fewshot_edge_index1 = torch.stack([head_vec, tail_vec], dim=0)
            fewshot_edge_index2 = torch.stack([tail_vec, head_vec], dim=0)
            fs_edge_index = torch.cat([fewshot_edge_index1, fewshot_edge_index2], dim=1).long().contiguous()
            fs_edge_type = torch.cat([torch.ones_like(head_vec) * target_rel, torch.ones_like(head_vec) * target_rel_inverse], dim=0)
            
            new_edge_index = torch.cat([new_edge_index, fs_edge_index], dim=1).long().contiguous()
            new_edge_type = torch.cat([new_edge_type, fs_edge_type], dim=0).long().contiguous()
            fseg_data = Data(
                edge_index= new_edge_index, 
                edge_type= new_edge_type,
                num_nodes=test_data.num_nodes, 
                num_relations=test_data.num_relations
            )
            fseg_data = tasks.build_relation_graph(fseg_data)
        else:
            fseg_data = zero_data

        if target_rel not in rel_cache_connections:
            fseg_edges = torch.cat([fseg_data.relation_graph.edge_index.t(), fseg_data.relation_graph.edge_type.unsqueeze(1)], dim=1)

            if target_rel not in enxrg_cache_data:
                # start_time = datetime.datetime.now()
                edge_mask = (fseg_edges[:,0] == target_rel) | (fseg_edges[:,1] == target_rel)
                unique_oth_edges = get_unique_connections(fseg_edges[~edge_mask], n_rel)

                lmeg_edges = torch.cat([test_data.relation_graph2.edge_index.t(), test_data.relation_graph2.edge_type.unsqueeze(1)], dim=1)
                edge_mask2 = (lmeg_edges[:,0] == target_rel) | (lmeg_edges[:,1] == target_rel)
                unique_lmrel_edges = get_unique_connections(lmeg_edges[edge_mask2], n_rel, ori_rel)
                unique_lmoth_edges = get_unique_connections(lmeg_edges[~edge_mask2], n_rel)

                if ds_name[:5]!="0shot":
                    unique_rel_edges = get_unique_connections(fseg_edges[edge_mask], n_rel, ori_rel)
                    new_connections = extend_rel_connections(ori_rel, unique_rel_edges, unique_oth_edges)
                    unique_candidate_edges = torch.cat([unique_lmrel_edges, new_connections], dim=0).unique(dim=0)
                else:
                    unique_candidate_edges = unique_lmrel_edges

                rel_edge0_keys_set = set(["_".join([str(h),str(t),str(r)]) for h,t,r in unique_oth_edges.detach().cpu().numpy().tolist()])
                rel_edge1_keys_set = set(["_".join([str(h),str(t),str(r)]) for h,t,r in unique_lmoth_edges.detach().cpu().numpy().tolist()])
                error_edge_set = list(rel_edge1_keys_set-rel_edge0_keys_set)

                if len(error_edge_set) != 0:
                    unique_error_edges = get_unique_connections(error_edge_set, n_rel)
                    remove_rel_dict = compute_remove_rels(unique_candidate_edges, unique_error_edges, ori_rel)
                    enxrg_cache_data[target_rel]=[unique_candidate_edges, remove_rel_dict]
                else:
                    enxrg_cache_data[target_rel]=[None, None]
            else:
                unique_candidate_edges, remove_rel_dict = enxrg_cache_data[target_rel]

            if unique_candidate_edges is not None:
                k = int(mode.split("_")[1])
                filtered_candidate_edges = remove_rels_from_candidates(unique_candidate_edges, remove_rel_dict, threshold=k)
            else:
                filtered_candidate_edges = unique_candidate_edges

            total_new_connections = build_inverse_connections(filtered_candidate_edges, n_rel)
            total_reledges = torch.cat([fseg_edges, total_new_connections], dim=0).unique(dim=0)
            rel_cache_connections[target_rel] = total_reledges
            # print("build graph:", (datetime.datetime.now()-start_time).total_seconds())
        else:
            total_reledges = rel_cache_connections[target_rel]

        rel_graph = Data(
            edge_index= total_reledges[:,(0,1)].t(),
            edge_type= total_reledges[:,2],
            edge_weight = torch.ones_like(total_reledges[:,2]),
            num_nodes=fseg_data.relation_graph.num_nodes,
            num_relations=4
        )
        fseg_data.relation_graph = rel_graph
        t_pred = model.forward(fseg_data, t_batch)
        h_pred = model.forward(fseg_data, h_batch)
        model.enxrg_cache_data = enxrg_cache_data
            
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
            
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    # ugly repetitive code for tail-only ranks processing
    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    # obtaining all ranks 
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    # the same for tails-only ranks
    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)
    
    metric_str = "%s_%s:" % (ds_name, mode)
    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking 
                _num_neg = all_num_negative 
                _metric_name = metric
            
            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            metric_str += "\t%s\t%.3f" % (metric, score)
            metrics[metric] = score.item()
    mrr = (1 / all_ranking.float()).mean()
    logger.warning(metric_str)
    return mrr if not return_metrics else metrics
 
#################################################### construct ########################################  
def build_rel_connections(rel_type_dict, pair_weight_dict=None, pair_ratio=0.5):
    rel_connections = []
    htype_dict = {}
    ttype_dict = {}
    for relid, value in rel_type_dict.items():
        value = {k.strip():v for k,v in value.items()}
        # head_list, tail_list = value["head"], value["tail"]
        if "head" in value:
            for item in value["head"]:
                if item not in htype_dict:
                    htype_dict[item] = set()
                htype_dict[item].add(relid)

        if "tail" in value:
            for item in value["tail"]:
                if item not in ttype_dict:
                    ttype_dict[item] = set()
                ttype_dict[item].add(relid)
    
    for key,vs in htype_dict.items():
        if len(vs) < 2: continue
        if pair_weight_dict is not None:
            if key in pair_weight_dict["h2h"]:
                weight = pair_weight_dict["h2h"][key]
                if weight < pair_ratio: 
                    continue
        combinations = list(itertools.combinations(vs, 2))
        rel_connections.extend([[h,"h2h",t, key] for h,t in combinations])
    for key,vs in ttype_dict.items():
        if len(vs) < 2: continue
        if pair_weight_dict is not None:
            if key in pair_weight_dict["t2t"]:
                weight = pair_weight_dict["t2t"][key]
                if weight < pair_ratio: 
                    continue
        combinations = list(itertools.combinations(vs, 2))
        rel_connections.extend([[h,"t2t",t, key] for h,t in combinations])

    for key1,vs1 in htype_dict.items():
        for key2,vs2 in ttype_dict.items():
            if key1 == key2:
                if pair_weight_dict is not None:
                    if key1 in pair_weight_dict["h2t"]:
                        weight = pair_weight_dict["h2t"][key1]
                        if weight < pair_ratio: 
                            continue
                combinations = [(item1, item2) for item1 in vs1 for item2 in vs2]
                rel_connections.extend([[h,"h2t",t, key1] for h,t in combinations])
    return rel_connections

def build_rel2type_relgraph(rel_connections, relation2id_dict, data_relid_list, n_rel):   
    # relgraph_mat[:,:,0] = np.eye(n_rel * 2) 
    # relgraph_mat[:,:,1] = np.eye(n_rel * 2) 
    # print(n_rel, len(relation2id_dict))
    type_dict= {"h2h":0, "t2t":1, "h2t":2, "t2h":3}
    relgraph_mat = np.zeros((n_rel * 2, n_rel * 2, len(type_dict)))
    # print(relgraph_mat.shape)
    
    for single_r in range(n_rel):
        if single_r in data_relid_list:
            relgraph_mat[single_r,single_r,0] = 1 # new added self-self connections
            relgraph_mat[single_r,single_r,1] = 1
            relgraph_mat[single_r+n_rel,single_r+n_rel,0] = 1
            relgraph_mat[single_r+n_rel,single_r+n_rel,1] = 1
            relgraph_mat[single_r,single_r+n_rel,2] = 1
            relgraph_mat[single_r,single_r+n_rel,3] = 1
            relgraph_mat[single_r+n_rel,single_r,2] = 1
            relgraph_mat[single_r+n_rel,single_r,3] = 1
        
    for fid, item in enumerate(rel_connections):
        h,r,t = item[:3]
        if h not in relation2id_dict.keys() or t not in relation2id_dict.keys():
            continue
        hid = relation2id_dict[h]
        tid = relation2id_dict[t]
        rid = type_dict[r]
        if isinstance(hid, torch.Tensor):
            hid = hid.item()
        if isinstance(tid, torch.Tensor):
            tid = tid.item()
        if hid not in data_relid_list or tid not in data_relid_list:
            continue
        # print(hid, tid, n_rel, hid+n_rel, tid+n_rel)
        if rid in [0, 1]: 
            if rid == 0: values = [0,1,2,3]
            elif rid == 1: values = [1,0,3,2]
            relgraph_mat[hid,tid,values[0]] = 1  # h2h when rid = 0
            relgraph_mat[tid,hid,values[0]] = 1
            relgraph_mat[hid+n_rel,tid+n_rel,values[1]] = 1  # t2t
            relgraph_mat[tid+n_rel,hid+n_rel,values[1]] = 1
            relgraph_mat[hid,tid+n_rel,values[2]] = 1 # h2t
            relgraph_mat[tid,hid+n_rel,values[2]] = 1
            relgraph_mat[hid+n_rel,tid,values[3]] = 1 # t2h
            relgraph_mat[tid+n_rel,hid,values[3]] = 1
        elif rid in [2, 3]:
            if rid == 2: values = [2,3,0,1]
            elif rid == 3: values = [3,2,1,0]
            relgraph_mat[hid,tid,values[0]] = 1  # h2t when rid = 2
            relgraph_mat[tid+n_rel,hid+n_rel,values[0]] = 1
            relgraph_mat[tid,hid,values[1]] = 1 # t2h
            relgraph_mat[hid+n_rel,tid+n_rel,values[1]] = 1  
            relgraph_mat[hid,tid+n_rel,values[2]] = 1 # h2h
            relgraph_mat[tid+n_rel,hid,values[2]] = 1
            relgraph_mat[hid+n_rel,tid,values[3]] = 1 # t2t
            relgraph_mat[tid,hid+n_rel,values[3]] = 1 
    return relgraph_mat
    
def build_relgraph_for_ds(rel_connections, single_data, device):
    relation2id_dict = {k:v for k,v in single_data.inv_rel_vocab}
    n_rel = int(len(relation2id_dict))
    data_relid_list = single_data.edge_type.long().unique().detach().cpu().numpy().tolist()
    relgraph_mat = build_rel2type_relgraph(rel_connections, relation2id_dict, data_relid_list, n_rel)
    
    indexes = np.nonzero(relgraph_mat)
    all_edge_index = torch.LongTensor(np.stack([indexes[0],indexes[1]],axis=0)).to(device)
    all_edge_type = torch.LongTensor(indexes[2]).to(device)
    # print(all_edge_index.shape, all_edge_type.shape)
    
    rel_graph = Data(
        edge_index=all_edge_index, 
        edge_type=all_edge_type,
        num_nodes=n_rel * 2, 
        num_relations=4
    ) 
    return rel_graph

def build_llm_relation_graph(data_list, file_root, llm_model="gpt4", input_mode="d&e_1", output_mode="fixed"):
    train_data, valid_data, test_data = data_list
    llm_model_name = llm_model
    input_mode_name = input_mode
    output_mode_name = output_mode
    reltype_ds = default_reltype_dict[graph]
    file_path = file_root + reltype_ds + "-r2t"
    llm_file_path = file_path+ "-" + llm_model_name + "-" + input_mode_name + "-" + output_mode_name+ ".json"
    # print(ds, llm_file_path)
    
    with open(llm_file_path, 'r') as f:
        rel_type_dict = json.load(f)
    new_rel_type_dict = {}
    for relid, value in rel_type_dict.items():
        value = {k.strip():v for k,v in value.items()}
        for side in ["head", "tail"]:
            if side in value:
                if len(value[side]) == 1 and isinstance(value[side][0], list): 
                    value[side] = value[side][0]
            if output_mode == "fixed":
                type_list = []
                if side in value:
                    for single_type in value[side]:
                        if single_type in candidate_reltype_list:
                            type_list.append(single_type)
                    value[side] = type_list
        new_rel_type_dict[relid] = value
    rel_type_dict = new_rel_type_dict
    
    rel_text_dict = {}
    with open(file_path+ ".txt", "r") as f:
        for line in f.readlines():
            rel, text = line.replace("\n","").split("\t")
            rel_text_dict[rel] = text
            
    train_rel_dict = dict(train_data.inv_rel_vocab)
    train_rel_set = set(train_rel_dict.keys())
    eval_rel_dict = dict(test_data.inv_rel_vocab)
    eval_rel_set = set(eval_rel_dict.keys())
    rel_connections = build_rel_connections(rel_type_dict)
    train_data.relation_graph2 = build_relgraph_for_ds(rel_connections, train_data, device)
    valid_data.relation_graph2 = build_relgraph_for_ds(rel_connections, valid_data, device)
    test_data.relation_graph2 = build_relgraph_for_ds(rel_connections, test_data, device)
    
    valid_data.rel_type_dict = rel_type_dict
    test_data.rel_type_dict = rel_type_dict
    return train_data, valid_data, test_data
        
if __name__ == "__main__":

    seeds = [1024, 42, 1337, 512, 256]

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-d", "--datasets", help="target datasets", default='FB15k237Inductive:v1,NELLInductive:v4', type=str, required=True)
    parser.add_argument("-g", "--gpu_num", help="the value same as --gpus", default=1, type=int)
    parser.add_argument("-reps", "--repeats", help="number of times to repeat each exp", default=1, type=int)
    parser.add_argument("-ft", "--finetune", help="finetune the checkpoint on the specified datasets", action='store_true')
    parser.add_argument("-tr", "--train", help="train the model from scratch", action='store_true')
    args, unparsed = parser.parse_known_args()
    print("gpus:", args.gpu_num)
    torch.cuda.set_device(args.gpu_num) 
   
    datasets = args.datasets.split(",")
    path = os.path.dirname(os.path.expanduser(__file__))
    results_file = os.path.join(path, f"ultra_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")

    for graph in datasets:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        for i in range(args.repeats):
            seed = seeds[i] if i < len(seeds) else random.randint(0, 10000)
            print(f"Running on {graph}, iteration {i+1} / {args.repeats}, seed: {seed}")

            # get dynamic arguments defined in the config file
            vars = util.detect_variables(args.config)
            parser = argparse.ArgumentParser()
            for var in vars:
                parser.add_argument("--%s" % var)
            vars = parser.parse_known_args(unparsed)[0]
            vars = {k: util.literal_eval(v) for k, v in vars._get_kwargs()}

            if args.finetune:
                epochs, batch_per_epoch = default_finetuning_config[ds] 
            elif args.train:
                epochs, batch_per_epoch = default_train_config[ds] 
            else:
                epochs, batch_per_epoch = 0, 'null'
            vars['epochs'] = epochs
            vars['bpe'] = batch_per_epoch
            vars['dataset'] = ds
            if version is not None:
                vars['version'] = version
            cfg = util.load_config(args.config, context=vars)

            root_dir = os.path.expanduser(cfg.output_dir) # resetting the path to avoid inf nesting
            os.chdir(root_dir)
            working_dir = util.create_working_directory(cfg)
            set_seed(seed)

            logger = util.get_root_logger()
            
            task_name = cfg.task["name"]
            dataset = util.build_dataset(cfg)
            device = util.get_device(cfg)
            
            train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
            
            train_data = tasks.build_relation_graph(train_data)
            valid_data = tasks.build_relation_graph(valid_data)
            test_data = tasks.build_relation_graph(test_data)

            train_data = tasks.build_relation_types(train_data)
            valid_data = tasks.build_relation_types(valid_data)
            test_data = tasks.build_relation_types(test_data)
            
            if cfg.model.entity_model["self_loop"]:
                train_data = tasks.build_selfloop_edges(train_data)
                valid_data = tasks.build_selfloop_edges(valid_data)
                test_data = tasks.build_selfloop_edges(test_data)
            
            train_data = train_data.to(device)
            valid_data = valid_data.to(device)
            test_data = test_data.to(device)

            if cfg.model["class"] == "Ultra":
                model = Ultra(
                    rel_model_cfg=cfg.model.relation_model,
                    entity_model_cfg=cfg.model.entity_model,
                )
            elif cfg.model["class"] == "Ultra2":
                model = Ultra2(
                    rel_model_cfg=cfg.model.relation_model,
                    entity_model_cfg=cfg.model.entity_model,
                )

            if "checkpoint" in cfg and cfg.checkpoint is not None:
                state = torch.load(cfg.checkpoint, map_location="cpu")
                model.load_state_dict(state["model"])
            model = model.to(device)
            
            if task_name == "InductiveInference":
                # filtering for inductive datasets
                # Grail, MTDEA, HM datasets have validation sets based off the training graph
                # ILPC, Ingram have validation sets from the inference graph
                # filtering dataset should contain all true edges (base graph + (valid) + test) 
                if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
                    # add inference, valid, test as the validation and test filtering graphs
                    full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
                    full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
                    test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
                    val_filtered_data = test_filtered_data
                else:
                    # test filtering graph: inference edges + test edges
                    full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
                    full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
                    test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

                    # validation filtering graph: train edges + validation edges
                    val_filtered_data = Data(
                        edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                        edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
                    )
            else:
                # for transductive setting, use the whole graph for filtered ranking
                filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
                val_filtered_data = test_filtered_data = filtered_data
            
            val_filtered_data = val_filtered_data.to(device)
            test_filtered_data = test_filtered_data.to(device)
            
            train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
                
            file_path = "../ingram_fewshot_ds/"
            llm_file_root = "../llmoutput/"
            total_metrics = []
            for llm_model in ["gpt4"]: # "llama7b", "llama13b", "mistral7b", "gpt3.5", "gpt4"
                for input_mode in ["des","exp_1","d&e_1"]:
                  for output_mode in ["fixed","refer","free"]:
                    train_data, valid_data, test_data = build_llm_relation_graph([train_data, valid_data, test_data],
                                                                                  llm_file_root, llm_model=llm_model, input_mode=input_mode, output_mode=output_mode)
                    for ds_version in ["v1", "v2", "v3"]:
                        for k_shot in ["3shot", "1shot", "0shot"]:
                            if k_shot == "0shot":
                                fs_ds_file = file_path + ds +"_"+ version + "-1shot-" + ds_version + ".json"
                            else:
                                fs_ds_file = file_path + ds +"_"+ version + "-" + k_shot + "-" + ds_version + ".json"
                            fs_ds_dict = json.load(open(fs_ds_file, 'r'))
                            model.enxrg_cache_data = {}
                            for mode in ["calib_-2", "calib_-1", "calib_1", "calib_3", "calib_5"]:
                                metrics = test_seq(cfg, model, valid_data, filtered_data=val_filtered_data, return_metrics=True, device=device, logger=logger, mode=mode, fs_ds_dict=fs_ds_dict, ds_name=k_shot + "-" + ds_version + "-" + llm_model + "-" + input_mode)
                                metrics.update({'dataset':graph, 'evaltype':"valid", 'mode':mode, 'k_shot':k_shot, 'llm_model':llm_model, 'prompt':input_mode, 'version':ds_version}.items())
                                total_metrics.append(metrics)

                                metrics = test_seq(cfg, model, test_data, filtered_data=test_filtered_data, return_metrics=True, device=device,
                                                    logger=logger, mode=mode, fs_ds_dict=fs_ds_dict, ds_name=k_shot + "-" + ds_version + "-" + llm_model + "-" + input_mode+ "-" + output_mode )
                                metrics.update({'dataset':graph, 'evaltype':"test", 'mode':mode, 'k_shot':k_shot, 'llm_model':llm_model, 'prompt':input_mode, 'enttype':output_mode, 'version':ds_version}.items())
                                total_metrics.append(metrics)
            
            # write to the log file
            with open(results_file, "a", newline='') as csv_file:
                fieldnames = ['dataset','mode']+list(metrics.keys()) #[:-1]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
                if csv_file.tell() == 0:
                    writer.writeheader()
                for metrics in total_metrics:
                    writer.writerow(metrics)