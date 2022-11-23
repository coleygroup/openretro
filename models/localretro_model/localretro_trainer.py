import logging
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from base.trainer_base import Trainer
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, EarlyStopping
from models.localretro_model.model import LocalRetro
from models.localretro_model.utils import load_dataloader, predict
from typing import Dict, List
from utils import misc


def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, atom_labels, bond_labels = batch_data
        if len(smiles) == 1:
            continue

        atom_labels, bond_labels = atom_labels.to(args.device), bond_labels.to(args.device)
        atom_logits, bond_logits, _ = predict(args, model, bg)

        loss_a = loss_criterion(atom_logits, atom_labels)
        loss_b = loss_criterion(bond_logits, bond_labels)
        total_loss = torch.cat([loss_a, loss_b]).mean()
        train_loss += total_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_clip)
        optimizer.step()

        if batch_id % args.print_every == 0:
            logging.info(f'\repoch {epoch + 1}/{args.num_epochs}, '
                         f'batch {batch_id + 1}/{len(data_loader)}, '
                         f'loss {total_loss: .4f}')

    logging.info(f'\nepoch {epoch + 1}/{args.num_epochs}, training loss: {train_loss / batch_id: .4f}')


def run_an_eval_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, atom_labels, bond_labels = batch_data
            atom_labels, bond_labels = atom_labels.to(args.device), bond_labels.to(args.device)
            atom_logits, bond_logits, _ = predict(args, model, bg)

            loss_a = loss_criterion(atom_logits, atom_labels)
            loss_b = loss_criterion(bond_logits, bond_labels)
            total_loss = torch.cat([loss_a, loss_b]).mean()
            val_loss += total_loss.item()
    return val_loss / batch_id


class LocalRetroTrainer(Trainer):
    """Class for LocalRetro Training"""

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AtomTemplate_n = len(pd.read_csv(
            os.path.join(self.processed_data_path, "atom_templates.csv")))
        self.BondTemplate_n = len(pd.read_csv(
            os.path.join(self.processed_data_path, "bond_templates.csv")))

        logging.info("Overwriting model args, based on original localretro training script")
        self.overwrite_model_args()
        misc.log_args(self.model_args, message="Updated model args")

    def overwrite_model_args(self):
        """Overwrite model args, adapted from Train.py"""
        # Need to pass some field into args; the source code always passes args around
        self.model_args.mode = "train"
        self.model_args.device = self.device
        self.model_args.processed_data_path = self.processed_data_path
        self.model_args.model_path = self.model_path
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                      'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                      'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                      'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba',
                      'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs']
        self.model_args.node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        self.model_args.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

    def build_train_model(self):
        args = self.model_args
        self.model = LocalRetro(
            node_in_feats=args.node_featurizer.feat_size(),
            edge_in_feats=args.edge_featurizer.feat_size(),
            node_out_feats=args.node_out_feats,
            edge_hidden_feats=args.edge_hidden_feats,
            num_step_message_passing=args.num_step_message_passing,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            AtomTemplate_n=self.AtomTemplate_n,
            BondTemplate_n=self.BondTemplate_n
        )
        self.model = self.model.to(self.device)

        logging.info("Logging model summary")
        logging.info(self.model)
        logging.info(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

    def train(self):
        """Core of Train.py"""
        args = self.model_args
        loss_criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.schedule_step)
        stopper = EarlyStopping(mode='lower', patience=args.patience, filename=self.model_path)

        train_loader, val_loader, test_loader = load_dataloader(args)
        for epoch in range(args['num_epochs']):
            run_a_train_epoch(args, epoch, self.model, train_loader, loss_criterion, optimizer)
            val_loss = run_an_eval_epoch(args, self.model, val_loader, loss_criterion)
            early_stop = stopper.step(val_loss, self.model)
            scheduler.step()
            logging.info(f'epoch {epoch + 1}/{args.num_epochs}, validation loss: {val_loss: .4f}')
            logging.info(f'epoch {epoch + 1}/{args.num_epochs}, Best loss: {stopper.best_score: .4f}')
            if early_stop:
                logging.info('Early stopped!!')
                break

        stopper.load_checkpoint(self.model)
        test_loss = run_an_eval_epoch(args, self.model, test_loader, loss_criterion)
        logging.info(f'test loss: {test_loss: .4f}')
