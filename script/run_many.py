import os
import sys
import csv
import math
import time
import pprint
import argparse
import random
import itertools

import numpy as np
import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra, Ultra2, Ultra3
from script.run import train_and_validate, test

   
torch.cuda.set_device(1) 
    
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


separator = ">" * 30
line = "-" * 30

def set_seed(seed):
    random.seed(seed + util.get_rank())
    # np.random.seed(seed + util.get_rank())
    torch.manual_seed(seed + util.get_rank())
    torch.cuda.manual_seed(seed + util.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    seeds = [1024, 42, 1337, 512, 256]

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-d", "--datasets", help="target datasets", default='FB15k237Inductive:v1,NELLInductive:v4', type=str, required=True)
    parser.add_argument("-reps", "--repeats", help="number of times to repeat each exp", default=1, type=int)
    parser.add_argument("-ft", "--finetune", help="finetune the checkpoint on the specified datasets", action='store_true')
    parser.add_argument("-tr", "--train", help="train the model from scratch", action='store_true')
    # parser.add_argument("-gpus", "--gpus", help="train the model from scratch", default='', type=str)
    args, unparsed = parser.parse_known_args()

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

            # args, vars = util.parse_args()
            # cfg = util.load_config(args.config, context=vars)
            # working_dir = util.create_working_directory(cfg)
            # torch.manual_seed(args.seed + util.get_rank())
            logger = util.get_root_logger()
            if util.get_rank() == 0:
                logger.warning("Random seed: %d" % seed)
                logger.warning("Config file: %s" % args.config)
                logger.warning(pprint.pformat(cfg))
            
            task_name = cfg.task["name"]
            dataset = util.build_dataset(cfg)
            device = util.get_device(cfg)
            
            train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]

            #############################################################################
            # load rel vocab by annual
            if ds == "CoDExMedium2": # _LLM
                # print(dir(train_data))
                # print(train_data.edge_index.shape, train_data.edge_type.shape)
                # print(train_data.edge_index[:,:10], train_data.edge_type[:10])
                # print(train_data.num_relations)
                data_file = "/home/kwang/git/ULTRA/kg-datasets/codex-m/raw/train.txt"
                n_rel = int(train_data.num_relations.item() / 2)
                relation2id_dict = {}
                with open(data_file, "r") as f:
                    for fid, line in enumerate(f.readlines()):
                        h,r,t = line.replace("\n","").split("\t")
                        # print(fid, h,r,t, train_data.edge_index[:,fid], train_data.edge_type[fid].item())
                        if r not in relation2id_dict.keys():
                            relation2id_dict[r] = train_data.edge_type[fid].item()
                        else:
                            assert relation2id_dict[r] == train_data.edge_type[fid].item()
                        if len(relation2id_dict) >= n_rel:
                            break
                # print(len(relation2id_dict), relation2id_dict)
                
                type_dict= {"h2h":0, "t2t":1, "h2t":2, "t2h":3}
                relgraph_mat = np.zeros((n_rel * 2, n_rel * 2, len(type_dict)))
                
                lemma_rel_connections = []
                llm_file = "/home/kwang/projects/ULTRA/llmoutput/m2_relgraph_18hop.txt" #
                with open(llm_file, "r") as f:
                    for fid, line in enumerate(f.readlines()):
                        h,r,t = line.replace("\n","").split(" ")
                        lemma_rel_connections.append([h,r,t])
                        
                
                gpt4_rel_connections = []
                from llmoutput.gpt4_enttype_dict import entity_types_analysis
                htype_dict = {}
                ttype_dict = {}
                for relid, value in entity_types_analysis.items():
                    head_list, tail_list = value["head"], value["tail"]
                    for item in head_list:
                        if item not in htype_dict:
                            htype_dict[item] = set()
                        htype_dict[item].add(relid)
                    for item in tail_list:
                        if item not in ttype_dict:
                            ttype_dict[item] = set()
                        ttype_dict[item].add(relid)
                
                for key,vs in htype_dict.items():
                    if len(vs) < 2: continue
                    combinations = list(itertools.combinations(vs, 2))
                    gpt4_rel_connections.extend([[h,"h2h",t] for h,t in combinations])
                for key,vs in ttype_dict.items():
                    if len(vs) < 2: continue
                    combinations = list(itertools.combinations(vs, 2))
                    gpt4_rel_connections.extend([[h,"t2t",t] for h,t in combinations])
                    
                for key1,vs1 in htype_dict.items():
                    for key2,vs2 in ttype_dict.items():
                        if key1 == key2:
                            combinations = [(item1, item2) for item1 in vs1 for item2 in vs2]
                            gpt4_rel_connections.extend([[h,"h2t",t] for h,t in combinations])
                
                
                # relgraph_mat[:,:,0] = np.eye(n_rel * 2) 
                # relgraph_mat[:,:,1] = np.eye(n_rel * 2) 
                
                for single_r in range(n_rel):
                    relgraph_mat[single_r,single_r+n_rel,2] = 1
                    relgraph_mat[single_r,single_r+n_rel,3] = 1
                    relgraph_mat[single_r+n_rel,single_r,2] = 1
                    relgraph_mat[single_r+n_rel,single_r,3] = 1
                    
                
                for fid, (h,r,t) in enumerate(gpt4_rel_connections):
                    if h not in relation2id_dict.keys() or t not in relation2id_dict.keys():
                        continue
                    hid = relation2id_dict[h]
                    tid = relation2id_dict[t]
                    rid = type_dict[r]
                    
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
                all_edge_index = torch.LongTensor(np.stack([indexes[0],indexes[1]],axis=0)).to(device)
                all_edge_type = torch.LongTensor(indexes[2]).to(device)
                print(all_edge_index.shape, all_edge_type.shape)
                
                rel_graph = Data(
                    edge_index=all_edge_index, 
                    edge_type=all_edge_type,
                    num_nodes=n_rel * 2, 
                    num_relations=4
                ) 

                train_data.relation_graph = rel_graph
                valid_data.relation_graph = rel_graph
                test_data.relation_graph = rel_graph
            elif "WikiTopicsMT2" in ds: # _LLM
                train_data = tasks.build_relation_graph(train_data)
                valid_data = tasks.build_relation_graph(valid_data)
                test_data = tasks.build_relation_graph(test_data)
              
            else:
                ### add for relation graph testing by KAI
                train_data = tasks.build_relation_graph(train_data)
                valid_data = tasks.build_relation_graph(valid_data)
                test_data = tasks.build_relation_graph(test_data)
                
                # rel_graph = train_data.relation_graph
                # valid_data.relation_graph = rel_graph
                # test_data.relation_graph = rel_graph

            train_data = tasks.build_relation_types(train_data)
            valid_data = tasks.build_relation_types(valid_data)
            test_data = tasks.build_relation_types(test_data)
            
            if cfg.model.entity_model["self_loop"]:
                train_data = tasks.build_selfloop_edges(train_data)
                valid_data = tasks.build_selfloop_edges(valid_data)
                test_data = tasks.build_selfloop_edges(test_data)
            ##############################################################################
            
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

            #model = pyg.compile(model, dynamic=True)
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
                #test_filtered_data = val_filtered_data = None
            else:
                # for transductive setting, use the whole graph for filtered ranking
                filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
                val_filtered_data = test_filtered_data = filtered_data
            
            val_filtered_data = val_filtered_data.to(device)
            test_filtered_data = test_filtered_data.to(device)
            
            train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Evaluate on valid")
            test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Evaluate on test")
            metrics = test(cfg, model, test_data, filtered_data=test_filtered_data, return_metrics=True, device=device, logger=logger)

            metrics = {k:v.item() for k,v in metrics.items()}
            metrics['dataset'] = graph
            # write to the log file
            with open(results_file, "a", newline='') as csv_file:
                fieldnames = ['dataset']+list(metrics.keys())[:-1]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
                if csv_file.tell() == 0:
                    writer.writeheader()
                writer.writerow(metrics)