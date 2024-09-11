import os
import sys
import copy
import math
import pprint
import random
import argparse
from itertools import islice
from functools import partial

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra, Ultra2


separator = ">" * 30
line = "-" * 30
os.environ["WORLD_SIZE"] = '3'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

def multigraph_collator_bak(batch, train_graphs, noeg_ratio = None):
    num_graphs = len(train_graphs)
    probs = torch.tensor([graph.edge_index.shape[1] for graph in train_graphs]).float()
    probs /= probs.sum()
    graph_id = torch.multinomial(probs, 1, replacement=False).item()

    graph = train_graphs[graph_id]
    bs = len(batch)
    edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]

    batch = torch.cat([graph.target_edge_index[:, edge_mask], graph.target_edge_type[edge_mask].unsqueeze(0)]).t()
    return graph, batch
    
def multigraph_collator(batch, train_graphs, balance_ratio = 0.5):
    num_graphs = len(train_graphs)
    probs = torch.tensor([graph.edge_index.shape[1] for graph in train_graphs]).float() # edge number
    # probs = torch.tensor([graph.num_nodes for graph in train_graphs]).float() # node number
    probs = torch.ones_like(probs) # modified for even training
    probs /= probs.sum() 
    graph_id = torch.multinomial(probs, 1, replacement=False).item()

    graph = train_graphs[graph_id]
    bs = len(batch)

    if random.random() > balance_ratio:
        edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]
        batch_type = 0
    else:
        # sample a relation id
        flag = True
        while flag:
            rid = random.randint(0, graph.num_relations//2)
            true_indices = torch.nonzero(graph.target_edge_type==rid, as_tuple=True)[0]
            if len(true_indices) > 2:
                flag = False
        edge_mask = true_indices[torch.randperm(true_indices.shape[0])[:bs]]
        batch_type = 1

    batch = torch.cat([graph.target_edge_index[:, edge_mask], graph.target_edge_type[edge_mask].unsqueeze(0)]).t()
    return graph, batch, batch_type


# here we assume that train_data and valid_data are tuples of datasets, add an additional dataset dict eval_graph_data for additioinal evaluation
def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None, batch_per_epoch=None, eval_graph_data=None,):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()
    
    pretrain_graph_names = cfg.dataset["graphs"]
    
    train_triplets = torch.cat([
        torch.cat([g.target_edge_index, g.target_edge_type.unsqueeze(0)]).t()
        for g in train_data
    ])
    
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    
    balance_ratio = 0.5 if cfg.train["fewshot_train"] < 1 else 0 #0.5
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler, collate_fn=partial(multigraph_collator, train_graphs=train_data, noeg_ratio=balance_ratio))

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                # now at each step we sample a new graph and edges from it
                train_graph, batch, batch_type = batch
                
                ########################################
                if cfg.train["fewshot_train"] < 1 and batch_type==1: # 0.5
                    b_rels = torch.unique(batch[:,2])
                    n_rel = train_graph.num_relations // 2
                    rel_mask = torch.ones_like(train_graph.edge_type).bool()
                    for target_rel in b_rels:
                        target_rel = target_rel.item()    
                        target_rel_inverse = target_rel + n_rel if target_rel < n_rel else target_rel - n_rel
                        relation_graph = train_graph.relation_graph
                        rel_mask1 = train_graph.edge_type != target_rel
                        rel_mask2 = train_graph.edge_type != target_rel_inverse
                        rel_mask = rel_mask & rel_mask1 & rel_mask2
                    
                    random_vec = (torch.rand(rel_mask.shape) <= cfg.train["fewshot_train"]).to(rel_mask.device)
                    rel_mask = rel_mask | random_vec
            
                    new_data = Data(
                        edge_index= train_graph.edge_index.t()[rel_mask].t(), 
                        edge_type= train_graph.edge_type[rel_mask],
                        num_nodes=train_graph.num_nodes, 
                        num_relations=train_graph.num_relations
                    )
                    new_train_graph = tasks.build_relation_graph(new_data)
                else:
                    new_train_graph = train_graph
                ########################################
                
                batch = tasks.negative_sampling(new_train_graph, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(new_train_graph, batch)
                
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
              
                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % (loss))
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data, graph_names=pretrain_graph_names)
        
        if eval_graph_data is not None:
            if rank == 0:
                logger.warning("Evaluate on test")
            test(cfg, model, eval_graph_data['test_data'], filtered_data=eval_graph_data["test_filter"], graph_names = eval_graph_data["graphs"])
        
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None, graph_names = None):
    world_size = util.get_world_size()
    rank = util.get_rank()
    # test_data is a tuple of validation/test datasets
    # process sequentially
    all_metrics = []
    for test_graph, filters, graph_name in zip(test_data, filtered_data, graph_names):
        metric_str = "\t %s: " % (graph_name)
        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

        model.eval()
        rankings = []
        num_negatives = []
        for batch in test_loader:
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            t_pred = model(test_graph, t_batch)
            h_pred = model(test_graph, h_batch)

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filters, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            rankings += [t_ranking, h_ranking]
            num_negatives += [num_t_negative, num_h_negative]

        ranking = torch.cat(rankings)
        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

        if rank == 0:
            for metric in cfg.task.metric:
                if metric == "mr":
                    score = all_ranking.float().mean()
                elif metric == "mrr":
                    score = (1 / all_ranking.float()).mean()
                elif metric.startswith("hits@"):
                    values = metric[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        # unbiased estimation
                        fp_rate = (all_ranking - 1).float() / all_num_negative
                        score = 0
                        for i in range(threshold):
                            # choose i false positive from num_sample - 1 negatives
                            num_comb = math.factorial(num_sample - 1) / \
                                    math.factorial(i) / math.factorial(num_sample - i - 1)
                            score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                        score = score.mean()
                    else:
                        score = (all_ranking <= threshold).float().mean()
                metric_str += "\t%s\t%.3f" % (metric, score)
        mrr = (1 / all_ranking.float()).mean()

        all_metrics.append(mrr)
        if rank == 0:
            # logger.warning(separator)
            logger.warning(metric_str)

    avg_metric = sum(all_metrics) / len(all_metrics)
    return avg_metric

def load_eval_graph(cfg, ds_list, eval_graph_data, mode="transductive"):
    for graph in ds_list:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        # get dynamic arguments defined in the config file
        vars = util.detect_variables(args.config)
        parser = argparse.ArgumentParser()
        for var in vars:
            parser.add_argument("--%s" % var)
        vars = parser.parse_known_args()[0] # unparsed
        vars = {k: util.literal_eval(v) for k, v in vars._get_kwargs()}
        epochs, batch_per_epoch = 0, 'null'
        vars['epochs'] = epochs
        vars['bpe'] = batch_per_epoch
        vars['dataset'] = ds
        if version is not None:
            vars['version'] = version
        cfg = util.load_config(args.config, context=vars)
        cfg.dataset['class'] = ds
        cfg.dataset['version'] = version
        cfg.dataset.pop('graphs')

        # root_dir = os.path.expanduser(cfg.output_dir) # resetting the path to avoid inf nesting
        # os.chdir(root_dir)
        # working_dir = util.create_working_directory(cfg)

        # logger = util.get_root_logger()
        # if util.get_rank() == 0:
        #     logger.warning("Config file: %s" % args.config)
        #     logger.warning(pprint.pformat(cfg))
        task_name = cfg.task["name"]
        dataset = util.build_dataset(cfg)
        device = util.get_device(cfg)
        train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
        valid_data = tasks.build_relation_graph(valid_data)
        test_data = tasks.build_relation_graph(test_data)
        
        if "fast_test" in cfg.train and mode == "transductive":
            num_val_edges = cfg.train.fast_test
            short_valid = copy.deepcopy(valid_data)
            mask = torch.randperm(short_valid.target_edge_index.shape[1])[:num_val_edges]
            short_valid.target_edge_index = short_valid.target_edge_index[:, mask]
            short_valid.target_edge_type = short_valid.target_edge_type[mask]
            valid_data = short_valid
        
        if mode == "inductive":
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
                    
        valid_data = valid_data.to(device)
        test_data = test_data.to(device)
        val_filtered_data = val_filtered_data.to(device)
        test_filtered_data = test_filtered_data.to(device)
        # eval_graph_data[graph] = {"dataset": ds, "version": version, "valid_data": valid_data, "test_data": test_data, "valid_filter":val_filtered_data, "test_filter":test_filtered_data}
        eval_graph_data["graphs"].append(graph)
        eval_graph_data["valid_data"].append(valid_data)
        eval_graph_data["test_data"].append(test_data)
        eval_graph_data["valid_filter"].append(val_filtered_data)
        eval_graph_data["test_filter"].append(test_filtered_data)
    return eval_graph_data


if __name__ == "__main__":
    args, vars = util.parse_args()
    
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    pretrain_graph_names = cfg.dataset["graphs"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset._data[0], dataset._data[1], dataset._data[2]
    
    eval_graph_data = {"graphs":[], "valid_data":[], "test_data":[], "valid_filter":[], "test_filter":[], }
    eval_graph_data = load_eval_graph(cfg, cfg.moreeval["evalgraphs"], eval_graph_data, mode="transductive")
    eval_graph_data = load_eval_graph(cfg, cfg.moreeval["ind_evalgraphs"], eval_graph_data, mode="inductive")

    if util.get_rank() == 0:
        logger.warning(eval_graph_data.keys())
    
    ##########################################################
    new_datas = []
    for data_list in [train_data, valid_data, test_data]:
        new_data_list = []
        for sd in data_list:
            sd = tasks.build_relation_graph(sd)
            new_data_list.append(sd)
        new_datas.append(new_data_list)
    train_data, valid_data, test_data = new_datas
    ############################################################
    
    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        if util.get_rank() == 0:
            logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
        short_valid = [copy.deepcopy(vd) for vd in valid_data]
        for graph in short_valid:
            mask = torch.randperm(graph.target_edge_index.shape[1])[:num_val_edges]
            graph.target_edge_index = graph.target_edge_index[:, mask]
            graph.target_edge_type = graph.target_edge_type[mask]
        
        short_valid = [sv.to(device) for sv in short_valid]

    train_data = [td.to(device) for td in train_data]
    valid_data = [vd.to(device) for vd in valid_data]
    test_data = [tst.to(device) for tst in test_data]

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

    if "checkpoint" in cfg:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)
    assert task_name == "MultiGraphPretraining", "Only the MultiGraphPretraining task is allowed for this script"
    
    # for transductive setting, use the whole graph for filtered ranking
    filtered_data = [
        Data(
            edge_index=torch.cat([trg.target_edge_index, valg.target_edge_index, testg.target_edge_index], dim=1), 
            edge_type=torch.cat([trg.target_edge_type, valg.target_edge_type, testg.target_edge_type,]),
            num_nodes=trg.num_nodes).to(device)
        for trg, valg, testg in zip(train_data, valid_data, test_data)
    ]
    
    train_and_validate(cfg, model, train_data, valid_data if "fast_test" not in cfg.train else short_valid, filtered_data=filtered_data, 
                       batch_per_epoch=cfg.train.batch_per_epoch, eval_graph_data = eval_graph_data)
                       
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data, graph_names = pretrain_graph_names)
    test(cfg, model, eval_graph_data['valid_data'], filtered_data=eval_graph_data["valid_filter"], graph_names = eval_graph_data["graphs"])
    
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data, graph_names = pretrain_graph_names)
    test(cfg, model, eval_graph_data['test_data'], filtered_data=eval_graph_data["test_filter"], graph_names = eval_graph_data["graphs"])