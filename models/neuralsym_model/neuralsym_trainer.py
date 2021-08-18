import logging
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from base.trainer_base import Trainer
from collections import defaultdict
from models.neuralsym_model.dataset import FingerprintDataset
from models.neuralsym_model.model import TemplateNN_Highway, TemplateNN_FC
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List


class NeuralSymTrainer(Trainer):
    """Class for NeuralSym Training"""

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

        self.templates_filtered = []
        self.init_templates()

    def init_templates(self):
        templates_file = os.path.join(self.processed_data_path, "training_templates.txt")
        logging.info(f'Loading templates from file: {templates_file}')
        with open(templates_file, 'r') as f:
            templates = f.readlines()
        for p in templates:
            pa, cnt = p.strip().split(': ')
            if int(cnt) >= self.model_args.min_freq:
                self.templates_filtered.append(pa)
        logging.info(f'Total number of template patterns: {len(self.templates_filtered) - 1}')

    def build_train_model(self):
        if self.model_args.model_arch == 'Highway':
            self.model = TemplateNN_Highway(
                output_size=len(self.templates_filtered),
                size=self.model_args.hidden_size,
                num_layers_body=self.model_args.depth,
                input_size=self.model_args.final_fp_size
            )
        elif self.model_args.model_arch == 'FC':
            self.model = TemplateNN_FC(
                output_size=len(self.templates_filtered),
                size=self.model_args.hidden_size,
                input_size=self.model_args.fp_size
            )
        else:
            raise ValueError(f"Unrecognized model name: {self.model_args.model_arch}")

        self.model = self.model.to(self.device)

        logging.info("Logging model summary")
        logging.info(self.model)
        logging.info(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

    def train(self):
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=self.model_args.learning_rate)

        train_dataset = FingerprintDataset(
            os.path.join(self.processed_data_path, "to_32681_prod_fps_train.npz"),
            os.path.join(self.processed_data_path, "labels_train.npy")
        )
        train_size = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=self.model_args.bs, shuffle=True)

        valid_dataset = FingerprintDataset(
            os.path.join(self.processed_data_path, "to_32681_prod_fps_val.npz"),
            os.path.join(self.processed_data_path, "labels_val.npy")
        )
        valid_size = len(valid_dataset)
        valid_loader = DataLoader(valid_dataset, batch_size=self.model_args.bs_eval, shuffle=False)
        del train_dataset, valid_dataset

        proposals_data_valid = pd.read_csv(
            os.path.join(self.processed_data_path, "processed_val.csv"),
            index_col=None, dtype='str'
        )

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',  # monitor top-1 val accuracy
            factor=self.model_args.lr_scheduler_factor,
            patience=self.model_args.lr_scheduler_patience,
            cooldown=self.model_args.lr_cooldown,
            verbose=True
        )

        train_losses, valid_losses = [], []
        k_to_calc = [1, 2, 3, 5, 10, 20, 50, 100]
        train_accs, val_accs = defaultdict(list), defaultdict(list)
        max_valid_acc = float('-inf')
        wait = 0  # early stopping patience counter
        start = time.time()
        for epoch in range(self.model_args.epochs):
            train_loss, train_correct, train_seen = 0, defaultdict(int), 0
            train_loader = tqdm(train_loader, desc='training')
            self.model.train()
            for data in train_loader:
                inputs, labels, idxs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.model.zero_grad()
                optimizer.zero_grad()
                outputs = self.model(inputs)
                # mask out rxn_smi w/ no valid template, giving loss = 0
                # logging.info(f'{outputs.shape}, {idxs.shape}, {(idxs < len(templates_filtered)).shape}')
                # [300, 10198], [300], [300]
                # Zhengkai: not really needed; just set ignore_index
                """
                outputs = torch.where(
                    (labels.view(-1, 1).expand_as(outputs) < len(self.templates_filtered)), outputs,
                    torch.tensor([0.], device=outputs.device)
                )
                labels = torch.where(
                    (labels < len(self.templates_filtered)), labels, torch.tensor([0.], device=labels.device).long()
                )
                """

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_seen += labels.shape[0]
                outputs = nn.Softmax(dim=1)(outputs)

                labels = torch.where(labels == 0, labels, torch.tensor([-1], device=labels.device))

                for k in k_to_calc:
                    batch_preds = torch.topk(outputs, k=k, dim=1)[1]
                    train_correct[k] += \
                        torch.where(batch_preds == labels.view(-1, 1).expand_as(batch_preds))[0].shape[0]

                train_loader.set_description(
                    f"training: loss={train_loss / train_seen:.4f}, "
                    f"top-1 acc={train_correct[1] / train_seen:.4f}")
                train_loader.refresh()
            train_losses.append(train_loss / train_seen)
            for k in k_to_calc:
                train_accs[k].append(train_correct[k] / train_seen)

            self.model.eval()
            with torch.no_grad():
                valid_loss, valid_correct, valid_seen = 0, defaultdict(int), 0
                valid_loader = tqdm(valid_loader, desc='validating')
                for i, data in enumerate(valid_loader):
                    inputs, labels, idxs = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item()
                    valid_seen += labels.shape[0]
                    outputs = nn.Softmax(dim=-1)(outputs)

                    labels = torch.where(labels == 0, labels, torch.tensor([-1], device=labels.device))
                    for k in k_to_calc:
                        batch_preds = torch.topk(outputs, k=k, dim=1)[1]
                        valid_correct[k] += \
                            torch.where(batch_preds == labels.view(-1, 1).expand_as(batch_preds))[0].shape[0]

                    valid_loader.set_description(
                        f"validating: loss={valid_loss/valid_seen:.4f}, "
                        f"top-1 acc={valid_correct[1] / valid_seen:.4f}")
                    valid_loader.refresh()

                    # display some examples + model predictions/labels for monitoring model generalization
                    try:
                        for j in range(i * self.model_args.bs_eval, (i + 1) * self.model_args.bs_eval):
                            # peek at a random sample of current batch to monitor training progress
                            if j % (valid_size // 5) == random.randint(0, 3) or j % (valid_size // 8) == random.randint(
                                    0, 4):
                                batch_preds = torch.topk(outputs, k=1)[1].squeeze(dim=-1)

                                rxn_idx = random.sample(list(range(self.model_args.bs_eval)), k=1)[0]
                                rxn_true_class = labels[rxn_idx]
                                rxn_pred_class = int(batch_preds[rxn_idx].item())
                                rxn_pred_score = outputs[rxn_idx, rxn_pred_class].item()
                                rxn_true_score = outputs[rxn_idx, rxn_true_class].item()

                                # load template database
                                rxn_pred_temp = self.templates_filtered[rxn_pred_class]
                                rxn_true_temp_idx = int(proposals_data_valid.iloc[idxs[rxn_idx].item(), 4])
                                if rxn_true_temp_idx < len(self.templates_filtered):
                                    rxn_true_temp = self.templates_filtered[rxn_true_temp_idx]
                                else:
                                    rxn_true_temp = 'Template not in training data'
                                rxn_true_prod = proposals_data_valid.iloc[idxs[rxn_idx].item(), 1]
                                rxn_true_prec = proposals_data_valid.iloc[idxs[rxn_idx].item(), 2]

                                # apply template to get predicted precursor,
                                # no need to reverse bcos alr: p_temp >> r_temp
                                rxn = rdchiralReaction(rxn_pred_temp)
                                prod = rdchiralReactants(rxn_true_prod)
                                rxn_pred_prec = rdchiralRun(rxn, prod)

                                logging.info(f'\ncurr product:                          \t\t{rxn_true_prod}')
                                logging.info(f'pred template:                          \t{rxn_pred_temp}')
                                logging.info(f'true template:                          \t{rxn_true_temp}')
                                if rxn_pred_class == len(self.templates_filtered):
                                    logging.info(f'pred precursor (score = {rxn_pred_score:+.4f}):\t\tNULL template')
                                elif len(rxn_pred_prec) == 0:
                                    logging.info(
                                        f'pred precursor (score = {rxn_pred_score:+.4f}):\t\tTemplate could not be applied')
                                else:
                                    logging.info(f'pred precursor (score = {rxn_pred_score:+.4f}):\t\t{rxn_pred_prec}')
                                logging.info(f'true precursor (score = {rxn_true_score:+.4f}):\t\t{rxn_true_prec}')
                                break
                    except Exception as e:
                        # do nothing # https://stackoverflow.com/questions/11414894/extract-traceback-info-from-an-exception-object/14564261#14564261
                        # tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                        # logging.info("".join(tb_str))
                        logging.info('\nIndex out of range (last minibatch)')

            # valid_losses.append(valid_loss/valid_seen)
            for k in k_to_calc:
                val_accs[k].append(valid_correct[k] / valid_seen)

            lr_scheduler.step(val_accs[1][-1])
            logging.info(f'\nCalled a step of ReduceLROnPlateau, current LR: {optimizer.param_groups[0]["lr"]}')

            if val_accs[1][-1] > max_valid_acc:
                # checkpoint model
                model_state_dict = self.model.state_dict()
                checkpoint_dict = {
                    "epoch": epoch,
                    "state_dict": model_state_dict, "optimizer": optimizer.state_dict(),
                    "train_accs": train_accs, "train_losses": train_losses,
                    "valid_accs": val_accs, "valid_losses": valid_losses,
                    "max_valid_acc": max_valid_acc
                }
                checkpoint_filename = os.path.join(self.model_path, f"{self.data_name}.pth.tar")
                torch.save(checkpoint_dict, checkpoint_filename)

            if self.model_args.early_stop and max_valid_acc - val_accs[1][-1] > self.model_args.early_stop_min_delta:
                if self.model_args.early_stop_patience <= wait:
                    message = f"\nEarly stopped at the end of epoch: {epoch}, \
                        \ntrain loss: {train_losses[-1]:.4f}, train top-1 acc: {train_accs[1][-1]:.4f}, \
                        \ntrain top-2 acc: {train_accs[2][-1]:.4f}, train top-3 acc: {train_accs[3][-1]:.4f}, \
                        \ntrain top-5 acc: {train_accs[5][-1]:.4f}, train top-10 acc: {train_accs[10][-1]:.4f}, \
                        \ntrain top-20 acc: {train_accs[20][-1]:.4f}, train top-50 acc: {train_accs[50][-1]:.4f}, \
                        \nvalid loss: N/A, valid top-1 acc: {val_accs[1][-1]:.4f} \
                        \nvalid top-2 acc: {val_accs[2][-1]:.4f}, valid top-3 acc: {val_accs[3][-1]:.4f}, \
                        \nvalid top-5 acc: {val_accs[5][-1]:.4f}, valid top-10 acc: {val_accs[10][-1]:.4f}, \
                        \nvalid top-20 acc: {val_accs[20][-1]:.4f}, valid top-50 acc: {val_accs[50][-1]:.4f}, \
                        \nvalid top-100 acc: {val_accs[100][-1]:.4f} \
                        \n"  # valid_losses[-1]:.4f}
                    logging.info(message)
                    break
                else:
                    wait += 1
                    logging.info(
                        f'\nIncrease in valid acc < early stop min delta {self.model_args.early_stop_min_delta}, \
                            \npatience count: {wait} \
                            \n'
                    )
            else:
                wait = 0
                max_valid_acc = max(max_valid_acc, val_accs[1][-1])

            message = f"\nEnd of epoch: {epoch}, \
                        \ntrain loss: {train_losses[-1]:.4f}, train top-1 acc: {train_accs[1][-1]:.4f}, \
                        \ntrain top-2 acc: {train_accs[2][-1]:.4f}, train top-3 acc: {train_accs[3][-1]:.4f}, \
                        \ntrain top-5 acc: {train_accs[5][-1]:.4f}, train top-10 acc: {train_accs[10][-1]:.4f}, \
                        \ntrain top-20 acc: {train_accs[20][-1]:.4f}, train top-50 acc: {train_accs[50][-1]:.4f}, \
                        \nvalid loss: N/A, valid top-1 acc: {val_accs[1][-1]:.4f} \
                        \nvalid top-2 acc: {val_accs[2][-1]:.4f}, valid top-3 acc: {val_accs[3][-1]:.4f}, \
                        \nvalid top-5 acc: {val_accs[5][-1]:.4f}, valid top-10 acc: {val_accs[10][-1]:.4f}, \
                        \nvalid top-20 acc: {val_accs[20][-1]:.4f}, valid top-50 acc: {val_accs[50][-1]:.4f}, \
                        \nvalid top-100 acc: {val_accs[100][-1]:.4f} \
                    \n"  # {valid_losses[-1]:.4f}
            logging.info(message)

        logging.info(f'Finished training, total time (minutes): {(time.time() - start) / 60}')
