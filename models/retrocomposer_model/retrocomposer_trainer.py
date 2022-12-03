import copy
import logging
import numpy as np
import os
import random
import torch
from base.trainer_base import Trainer
from models.retrocomposer_model.gnn import GNN_graphpred
from models.retrocomposer_model.prepare_mol_graph import MoleculeDataset
from torch import optim
from torch_geometric.data import DataLoader
from tqdm import tqdm
from typing import Dict, List


def _train(args, model, device, loader, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss_list, loss_prod_list, loss_react_list = [], [], []
    prod_pred_res_max, react_pred_res, react_pred_res_each = [], [], []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        loss_prod, loss_react, prod_pred_max, react_pred = model(batch)
        loss = loss_prod + loss_react
        loss_list.append(loss.item())
        loss_prod_list.append(loss_prod.item())
        loss_react_list.append(loss_react.item())
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        prod_pred_res_max.extend(prod_pred_max)
        react_pred_res_each.extend(react_pred.reshape(-1, ).tolist())
        for react in react_pred:
            react_pred_res.append(False not in react)

    loss = np.mean(loss_list)
    loss_prod = np.mean(loss_prod_list)
    loss_react = np.mean(loss_react_list)
    prod_pred_acc_max = np.mean(prod_pred_res_max)
    react_pred_acc_each = np.mean(react_pred_res_each)
    react_pred_acc = np.mean(react_pred_res)

    return loss, loss_prod, loss_react, prod_pred_acc_max, react_pred_acc_each, react_pred_acc


def _train_multiprocess(rank, args, model, device, train_dataset, val_dataset):
    logging.info("Building optimizer and scheduler")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    epochs = args.epochs // args.num_process
    output_model_file = os.path.join(args.model_path, 'model.pt')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset_ = copy.deepcopy(val_dataset)
    valid_dataset_.processed_data_files = valid_dataset_.processed_data_files_valid
    val_loader = DataLoader(valid_dataset_, batch_size=args.batch_size, shuffle=False)
    for epoch in range(1, epochs + 1):
        logging.info("====rank and epoch: ", rank, epoch)
        train_res = _train(args, model, device, train_loader, optimizer)
        loss, loss_prod, loss_react, prod_pred_acc_max, react_pred_acc_each, react_pred_acc = train_res
        logging.info(f"rank: {rank} epoch: {epoch} train loss: {loss} loss_prod: {loss_prod} loss_react: {loss_react} "
                     f"prod_pred_acc_max: {prod_pred_acc_max} "
                     f"react_pred_acc_each: {react_pred_acc_each} "
                     f"react_pred_acc: {react_pred_acc}")
        scheduler.step()
        if rank == 0:
            torch.save(model.state_dict(), output_model_file)
        logging.info("====evaluation")
        val_res = _train(args, model, device, val_loader, train=False)
        loss, loss_prod, loss_react, prod_pred_acc_max, react_pred_acc_each, react_pred_acc = val_res
        logging.info(f"epoch: {epoch} validation loss: {loss} loss_prod: {loss_prod} loss_react: {loss_react} "
                     f"prod_pred_acc_max: {prod_pred_acc_max} "
                     f"react_pred_acc_each: {react_pred_acc_each} "
                     f"react_pred_acc: {react_pred_acc}")


class RetroComposerTrainerS1(Trainer):
    """Class for RetroComposer Training, Stage 1"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 model_path: str):
        super().__init__(model_name=model_name,
                         model_args=model_args,
                         model_config=model_config,
                         data_name=data_name,
                         processed_data_path=processed_data_path,
                         model_path=model_path)

        random.seed(self.model_args.seed)
        np.random.seed(self.model_args.seed)
        torch.manual_seed(self.model_args.seed)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() and not model_args.use_cpu
                                   else "cpu")
        self.model_args.model_path = self.model_path
        self.train_dataset = MoleculeDataset(
            root=processed_data_path, split='train', load_mol=True)
        self.val_dataset = MoleculeDataset(
            root=processed_data_path, split='val', load_mol=False)
        self.prod_word_size = len(self.train_dataset.prod_smarts_fp_list)
        self.react_word_size = len(self.train_dataset.react_smarts_list)

    def build_train_model(self):
        args = self.model_args
        self.model = GNN_graphpred(
            args.num_layer,
            args.emb_dim,
            args.atom_feat_dim,
            args.bond_feat_dim,
            args.center_loss_type,
            0,
            self.prod_word_size,
            self.react_word_size,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            graph_pooling=args.graph_pooling
        )
        del self.model.gnn_diff
        del self.model.scoring
        self.model = self.model.to(self.device)

        logging.info("Logging model summary")
        logging.info(self.model)
        logging.info(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

    def train(self):
        """Core of run_retro.py"""
        args = self.model_args
        output_model_file = os.path.join(self.model_path, "model.pt")

        if args.multiprocess:
            logging.info(f"Start training in multi-process mode. num_process = {args.num_process}")
            torch.multiprocessing.set_start_method('spawn', force=True)
            self.model.share_memory()  # gradients are allocated lazily, so they are not shared here
            processes = []
            for rank in range(args.num_process):
                p = torch.multiprocessing.Process(
                    target=_train_multiprocess,
                    args=(rank, args, self.model, self.device, self.train_dataset, self.val_dataset)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.close()
                p.join()
        else:
            logging.info("Start training in single-process mode")

            logging.info("Building optimizer and scheduler")
            optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
            logging.info(optimizer)

            train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
            self.val_dataset.processed_data_files = self.val_dataset.processed_data_files_valid
            val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False)

            for epoch in range(1, args.epochs + 1):
                logging.info("====epoch " + str(epoch))
                res = _train(args, self.model, self.device, train_loader, optimizer)
                loss, loss_prod, loss_react, prod_pred_acc_max, react_pred_acc_each, react_pred_acc = res
                logging.info(f"epoch: {epoch} train loss: {loss} loss_prod: {loss_prod} loss_react: {loss_react} "
                             f"prod_pred_acc_max: {prod_pred_acc_max} "
                             f"react_pred_acc_each: {react_pred_acc_each} "
                             f"react_pred_acc: {react_pred_acc}")
                scheduler.step()
                torch.save(self.model.state_dict(), output_model_file)

                logging.info("====evaluation")
                val_res = _train(args, self.model, self.device, val_loader, train=False)
                loss, loss_prod, loss_react, prod_pred_acc_max, react_pred_acc_each, react_pred_acc = val_res
                logging.info(f"epoch: {epoch} validation loss: {loss} loss_prod: {loss_prod} loss_react: {loss_react} "
                             f"prod_pred_acc_max: {prod_pred_acc_max} "
                             f"react_pred_acc_each: {react_pred_acc_each} "
                             f"react_pred_acc: {react_pred_acc}")
