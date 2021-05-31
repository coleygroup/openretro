import dgl
import json
import logging
import numpy as np
import os
import random
import torch
import torch.nn as nn
from base.trainer_base import Trainer
from models.retroxpert_model.data import RetroCenterDatasets
from models.retroxpert_model.model.gat import GATNet
from onmt.bin.train import train as onmt_train
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List


def collate(_data):
    return map(list, zip(*_data))


class RetroXpertTrainerS1(Trainer):
    """Class for RetroXpert Training, Stage 1"""

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
        self.milestones = []
        self.device = torch.device("cuda" if torch.cuda.is_available() and not model_args.use_cpu
                                   else "cpu")
        if self.model_args.typed:
            self.model_args.in_dim += 10
            self.exp_name = f"{data_name}_typed"
        else:
            self.exp_name = f"{data_name}_untyped"

        self.meta_data_file = os.path.join(processed_data_path, "metadata.json")
        with open(self.meta_data_file, "r") as f:
            self.meta_data = json.load(f)

        self.semi_template_count = self.meta_data["semi_template_count"]

        self.checkpoint_path = os.path.join(self.model_path, f"{self.exp_name}_checkpoint.pt")

    def build_train_model(self):
        self.model = GATNet(
            in_dim=self.model_args.in_dim + self.semi_template_count,
            num_layers=self.model_args.gat_layers,
            hidden_dim=self.model_args.hidden_dim,
            heads=self.model_args.heads,
            use_gpu=(not self.model_args.use_cpu),
        )
        self.model = self.model.to(self.device)

        logging.info("Logging model summary")
        logging.info(self.model)
        logging.info(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

        if self.model_args.load_checkpoint_s1 and os.path.exists(self.checkpoint_path):
            logging.info(f"Loading from {self.checkpoint_path}")

            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            self.model_args.lr *= 0.2
            self.milestones = []
        else:
            self.milestones = [20, 40, 60, 80]

    def train(self):
        """Core of train.py"""
        logging.info("Creating optimizer")
        optimizer = torch.optim.Adam([{'params': self.model.parameters()}],
                                     lr=self.model_args.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=0.2)

        logging.info("Creating data loaders")
        train_data = RetroCenterDatasets(processed_data_path=self.processed_data_path,
                                         fns=["rxn_data_train.pkl", "pattern_feat_train.npz"])
        train_dataloader = DataLoader(train_data,
                                      batch_size=self.model_args.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=collate)

        valid_data = RetroCenterDatasets(processed_data_path=self.processed_data_path,
                                         fns=["rxn_data_val.pkl", "pattern_feat_val.npz"])
        valid_dataloader = DataLoader(valid_data,
                                      batch_size=4*self.model_args.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      collate_fn=collate)

        logging.info("Start training")
        for epoch in range(1, 1 + self.model_args.epochs):
            self.model.train()
            total = 0.
            correct = 0.
            epoch_loss = 0.
            epoch_loss_ce = 0.
            epoch_loss_h = 0.

            progress_bar = tqdm(train_dataloader)
            for i, data in enumerate(progress_bar):
                progress_bar.set_description(f"Epoch {epoch}")
                rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj, disconnection_num = data

                x_atom = list(map(lambda x: torch.from_numpy(x).float(), x_atom))
                x_pattern_feat = list(map(lambda x: torch.from_numpy(x).float(), x_pattern_feat))
                x_atom = list(map(lambda x, y: torch.cat([x, y], dim=1), x_atom, x_pattern_feat))

                if self.model_args.typed:
                    rxn_class = list(map(lambda x: torch.from_numpy(x).float(), rxn_class))
                    x_atom = list(map(lambda x, y: torch.cat([x, y], dim=1), x_atom, rxn_class))

                x_atom = torch.cat(x_atom, dim=0)
                disconnection_num = torch.LongTensor(disconnection_num)

                x_atom = x_atom.to(self.device)
                disconnection_num = disconnection_num.to(self.device)

                x_adj = list(map(lambda x: torch.from_numpy(np.array(x)), x_adj))
                y_adj = list(map(lambda x: torch.from_numpy(np.array(x)), y_adj))
                x_adj = [xa.to(self.device) for xa in x_adj]
                y_adj = [ye.to(self.device) for ye in y_adj]

                mask = list(map(lambda x: x.view(-1, 1).bool(), x_adj))
                bond_connections = list(map(lambda x, y: torch.masked_select(x.reshape(-1, 1), y), y_adj, mask))
                bond_labels = torch.cat(bond_connections, dim=0).float()

                self.model.zero_grad()

                # batch graph
                g_dgl = dgl.batch(x_graph)
                h_pred, e_pred = self.model(g_dgl, x_atom)
                e_pred = e_pred.squeeze()
                loss_h = nn.CrossEntropyLoss(reduction="sum")(h_pred, disconnection_num)
                loss_ce = nn.BCEWithLogitsLoss(reduction="sum")(e_pred, bond_labels)
                loss = loss_ce + loss_h
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_ce += loss_ce.item()
                epoch_loss_h += loss_h.item()

                start = end = 0
                pred = torch.round(torch.sigmoid(e_pred)).long()
                edge_lens = list(map(lambda x: x.shape[0], bond_connections))
                cur_batch_size = len(edge_lens)
                bond_labels = bond_labels.long()
                for j in range(cur_batch_size):
                    start = end
                    end += edge_lens[j]
                    if torch.equal(pred[start:end], bond_labels[start:end]):
                        correct += 1
                assert end == len(pred)

                total += cur_batch_size
                progress_bar.set_postfix(
                    loss="%.5f" % (epoch_loss / total),
                    acc="%.5f" % (correct / total),
                    loss_ce="%.5f" % (epoch_loss_ce / total),
                    loss_h="%.5f" % (epoch_loss_h / total)
                )

            scheduler.step(epoch)
            train_acc = correct / total
            train_loss = epoch_loss / total
            logging.info(f"End of epoch {epoch}. Train Loss: {train_loss: .5f}, "
                         f"Train Bond Disconnection Acc: {train_acc: .5f}")

            if epoch % 5 == 0:
                valid_acc = self.test(valid_dataloader, data_split="val", save_pred=False)
                logging.info(f"Epoch {epoch}, train_acc: {train_acc}, "
                             f"valid_acc: {valid_acc}, train_loss: {train_loss}")
                torch.save(self.model.state_dict(), self.checkpoint_path)

        torch.save(self.model.state_dict(), self.checkpoint_path)

        self.average_models()       # TODO

    def average_models(self):
        """Averaging model checkpoints"""
        pass

    def test(self, dataloader: DataLoader, data_split: str = "", save_pred=False):
        """Adapted from train.test()"""
        self.model.eval()
        correct = 0.
        total = 0.
        epoch_loss = 0.
        # Bond disconnection probability
        pred_true_list = []
        pred_logits_mol_list = []
        # Bond disconnection number gt and prediction
        bond_change_gt_list = []
        bond_change_pred_list = []

        for i, data in enumerate(tqdm(dataloader)):
            rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj, disconnection_num = data

            x_atom = list(map(lambda x: torch.from_numpy(x).float(), x_atom))
            x_pattern_feat = list(map(lambda x: torch.from_numpy(x).float(), x_pattern_feat))
            x_atom = list(map(lambda x, y: torch.cat([x, y], dim=1), x_atom, x_pattern_feat))

            if self.model_args.typed:
                rxn_class = list(map(lambda x: torch.from_numpy(x).float(), rxn_class))
                x_atom = list(map(lambda x, y: torch.cat([x, y], dim=1), x_atom, rxn_class))

            x_atom = torch.cat(x_atom, dim=0)
            disconnection_num = torch.LongTensor(disconnection_num)

            x_atom = x_atom.to(self.device)
            disconnection_num = disconnection_num.to(self.device)

            x_adj = list(map(lambda x: torch.from_numpy(np.array(x)), x_adj))
            y_adj = list(map(lambda x: torch.from_numpy(np.array(x)), y_adj))
            x_adj = [xa.to(self.device) for xa in x_adj]
            y_adj = [ye.to(self.device) for ye in y_adj]

            mask = list(map(lambda x: x.view(-1, 1).bool(), x_adj))
            bond_disconnections = list(map(lambda x, y: torch.masked_select(x.reshape(-1, 1), y), y_adj, mask))
            bond_labels = torch.cat(bond_disconnections, dim=0).float()

            # batch graph
            g_dgl = dgl.batch(x_graph)
            h_pred, e_pred = self.model(g_dgl, x_atom)
            e_pred = e_pred.squeeze()
            loss_h = nn.CrossEntropyLoss(reduction="sum")(h_pred, disconnection_num)
            loss_ce = nn.BCEWithLogitsLoss(reduction="sum")(e_pred, bond_labels)
            loss = loss_ce + loss_h
            epoch_loss += loss.item()

            h_pred = torch.argmax(h_pred, dim=1)
            bond_change_pred_list.extend(h_pred.cpu().tolist())
            bond_change_gt_list.extend(disconnection_num.cpu().tolist())

            start = end = 0
            pred = torch.sigmoid(e_pred)
            edge_lens = list(map(lambda x: x.shape[0], bond_disconnections))
            cur_batch_size = len(edge_lens)
            bond_labels = bond_labels.long()

            for j in range(cur_batch_size):
                start = end
                end += edge_lens[j]
                label_mol = bond_labels[start:end]
                pred_proab = pred[start:end]
                mask_pos = torch.nonzero(x_adj[j]).tolist()
                assert len(mask_pos) == len(pred_proab)

                pred_disconnection_adj = torch.zeros_like(x_adj[j], dtype=torch.float32)
                for idx, pos in enumerate(mask_pos):
                    pred_disconnection_adj[pos[0], pos[1]] = pred_proab[idx]
                for idx, pos in enumerate(mask_pos):
                    pred_proab[idx] = (pred_disconnection_adj[pos[0], pos[1]] +
                                       pred_disconnection_adj[pos[1], pos[0]]) / 2

                pred_mol = pred_proab.round().long()
                if torch.equal(pred_mol, label_mol):
                    correct += 1
                    pred_true_list.append(True)
                    pred_logits_mol_list.append([
                        True,
                        label_mol.tolist(),
                        pred_proab.tolist(),
                    ])
                else:
                    pred_true_list.append(False)
                    pred_logits_mol_list.append([
                        False,
                        label_mol.tolist(),
                        pred_proab.tolist(),
                    ])
                total += 1

        pred_lens_true_list = list(map(lambda x, y: x == y, bond_change_gt_list, bond_change_pred_list))
        bond_change_pred_list = list(map(lambda x, y: [x, y], bond_change_gt_list, bond_change_pred_list))
        acc = correct / total

        logging.info(f"Bond disconnection number prediction acc: {np.mean(pred_lens_true_list): .6f}")
        logging.info(f"Loss: {epoch_loss / total}")
        logging.info(f"Bond disconnection acc (without auxiliary task): {acc: .6f}")

        if save_pred:
            logging.info(f"pred_true_list size: {len(pred_true_list)}")
            disconnection_fn = os.path.join(
                self.processed_data_path, f"{data_split}_disconnection_{self.exp_name}.txt")
            result_fn = os.path.join(
                self.processed_data_path, f"{data_split}_result_{self.exp_name}.txt")
            result_mol_fn = os.path.join(
                self.processed_data_path, f"{data_split}_result_mol_{self.exp_name}.txt")

            logging.info("Saving prediction results")
            np.savetxt(disconnection_fn, np.asarray(bond_change_pred_list), fmt="%d")
            np.savetxt(result_fn, np.asarray(pred_true_list), fmt="%d")
            with open(result_mol_fn, "w") as f:
                for i, line in enumerate(pred_logits_mol_list):
                    f.write(f"{i} {line[0]}\n")
                    f.write(" ".join([str(i) for i in line[1]]) + "\n")
                    f.write(" ".join([str(i) for i in line[2]]) + "\n")

        return acc


class RetroXpertTrainerS2(Trainer):
    """Class for RetroXpert Training, Stage 2"""

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

        logging.info("Overwriting model args, (hardcoding essentially)")
        self.overwrite_model_args()
        logging.info(f"Updated model args: {self.model_args}")

    def overwrite_model_args(self):
        """Overwrite model args"""
        # Paths
        self.model_args.data = os.path.join(self.processed_data_path, "bin")
        self.model_args.save_model = os.path.join(self.model_path, "model")

    def build_train_model(self):
        logging.info("For onmt training, models are built implicitly.")

    def train(self):
        """A wrapper to onmt.bin.train()"""
        onmt_train(self.model_args)

    def test(self):
        """TBD"""
        raise NotImplementedError

