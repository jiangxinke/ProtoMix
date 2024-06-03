import os
import logging
import torch
import random
import numpy as np
import json
import torch.optim as optim

from model.ProtoMix import ProtoMix
from time import time
from model.run_utils import calc_label_freq, prototyping, get_metrics_fn, get_dataloader
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from parser import parse_args
import pickle
from sklearn.model_selection import train_test_split

def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):

    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All logs will be saved to %s" %logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def patient_train_val_test_split(dataset, ratios, seed):
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(range(0, len(dataset), 1))
    label_list = [sample["label"] for sample in dataset]
    temp_index, test_index, y_temp, y_test = \
        train_test_split(patient_indx, label_list, test_size=ratios[2], stratify=label_list, random_state=seed)
    train_index, val_index, y_train, y_val = train_test_split(temp_index, y_temp,
                                                              test_size=ratios[1] / ratios[0] + ratios[1],
                                                              stratify=y_temp,
                                                              random_state=seed)

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return score > best_score
    elif monitor_criterion == "min":
        return score < best_score
    else:
        raise ValueError(f"Monitor criterion {monitor_criterion} is not supported")


def inference(model, dataloader, additional_outputs=None) -> Dict[str, float]:
    loss_all = []
    y_true_all = []
    y_prob_all = []
    patient_embed_all = []
    if additional_outputs is not None:
        additional_outputs = {k: [] for k in additional_outputs}

    for _,data in enumerate(dataloader):
        model.eval()
        with torch.no_grad():
            loss_origincls, origin_cls_logits, patient_y_true, patient_y_prob, patient_embed = model("test",**data)
            y_true = patient_y_true.data.cpu().numpy()
            y_prob = patient_y_prob.data.cpu().numpy()
            loss_all.append(loss_origincls.item())
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            patient_embed_all.append(patient_embed.data.cpu().numpy())

    loss_mean = sum(loss_all) / len(loss_all)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    patient_embed_all = np.vstack(patient_embed_all)
    if additional_outputs is not None:
        additional_outputs = {key: np.concatenate(val)
                              for key, val in additional_outputs.items()}
        return y_true_all, y_prob_all, loss_mean, additional_outputs
    return y_true_all, y_prob_all, loss_mean, patient_embed_all

def evaluate(model, dataloader, metrics) -> Dict[str, float]:
    y_true_all, y_prob_all, loss_mean, patient_embed_all = inference(model, dataloader)
    mode = model.model.mode
    metrics_fn = get_metrics_fn(mode)
    scores = metrics_fn(y_true_all, y_prob_all, metrics=metrics)
    scores["loss"] = loss_mean
    return scores, patient_embed_all, y_true_all

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.dataset == "MIMIC3":
        datasample = load_pickle("../sample_dataset_mimiciii_mortality.pkl")
        x_key = ["medical_code"]

    elif args.dataset == "MIMIC4":
        datasample = load_pickle("../sample_dataset_mimiciv_mortality.pkl")
        x_key = ["medical_code"]

    train_ds, val_ds, test_ds = patient_train_val_test_split(datasample, eval(args.train_val_test_split), args.seed)

    positive_num_train, negative_num_train, positive_frequency, negative_frequency = calc_label_freq(train_ds)
    logging.info("Positive_num_train: ", positive_num_train)
    logging.info("Negative_num_train: ", negative_num_train)
    logging.info("Positive_frequency: {:.4f}".format(positive_frequency))
    logging.info("Negative_frequency: {:.4f}".format(negative_frequency))

    train_dataloader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dataloader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    model_kwargs = {"dataset": datasample, "feature_keys": x_key, "label_key": "label", "mode": "binary",
                    "train_patient_num": len(train_ds), "positive_frequency": positive_frequency,
                    "negative_frequency": negative_frequency}
    model = ProtoMix(args, **model_kwargs)

    model.to(device)
    logging.info(model)
    logging.info(args.cuda_choice)
    with open(args.save_dir + "params.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    last_checkpoint_path_name = os.path.join(args.save_dir, "last.ckpt")
    best_checkpoint_path_name = os.path.join(args.save_dir, "best.ckpt")

    if args.use_last_checkpoint != -1:
        logging.info(f"Loading checkpoint from {last_checkpoint_path_name}")
        state_dict = torch.load(last_checkpoint_path_name, map_location=args.cuda_choice)
        model.load_state_dict(state_dict)
    logging.info("")

    logging.info("Training:")
    param = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in param if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer_train = optim.Adam(optimizer_grouped_parameters, lr = args.train_lr)

    data_iterator = iter(train_dataloader)
    best_score = -1 * float("inf") if args.monitor_criterion == "max" else float("inf")
    steps_per_epoch = len(train_dataloader)
    global_step = 0
    best_dev_epoch = 0

    for epoch in range(args.epochs_train):
        time0 = time()
        training_loss_protolabel = []
        training_loss_protocls = []
        training_loss_origincls = []
        training_loss_all = []
        patient_embedding_train = []

        model.train()

        for _ in range(steps_per_epoch):
            optimizer_train.zero_grad()
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_dataloader)
                data = next(data_iterator)

            loss_protolabel, loss_protocls, loss_origincls, patient_y_true, patient_y_prob, patient_embed = model(
                "train", **data)
            loss_all = args.alpha * loss_protolabel + (1.0-args.alpha) * loss_protocls + loss_origincls
            loss_all.backward()

            patient_embedding_train.append(patient_embed.clone().detach())

            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer_train.step()

            training_loss_protolabel.append(loss_protolabel.item())
            training_loss_protocls.append(loss_protocls.item())
            training_loss_origincls.append(loss_origincls.item())
            training_loss_all.append(loss_all.item())
            global_step += 1

        patient_embedding_train_all = torch.cat(patient_embedding_train, dim=0)
        model.patientEmbedUpdate(patient_embedding_train_all)

        logging.info("--- Train epoch-{}, step-{}, Total Time {:.1f}s---".format(epoch, global_step, time() - time0))
        logging.info(f"loss_all: {sum(training_loss_all) / len(training_loss_all):.4f}")
        logging.info(f"loss_protolabel: {sum(training_loss_protolabel) / len(training_loss_protolabel):.4f}")
        logging.info(f"loss_protocls: {sum(training_loss_protocls) / len(training_loss_protocls):.4f}")
        logging.info(f"loss_origincls: {sum(training_loss_origincls) / len(training_loss_origincls):.4f}")

        logging.info(f"--- Prototyping ---")
        time0 = time()
        prototype_labels, centroids = prototyping(model.train_patient_embed.clone().detach(), args.num_prototypes, max_iters=1000,
                                            tol=1e-4, alpha=args.prototype_alpha)
        model.prototypeEmbedUpdate(centroids)
        logging.info(
                '--- Prototyping Total Time {:.1f}s ---'.format(time()-time0))

        if last_checkpoint_path_name is not None:
            state_dict = model.state_dict()
            torch.save(state_dict, last_checkpoint_path_name)

        if val_dataloader is not None:
            scores, patient_embed_all, y_true_all = evaluate(model, val_dataloader, args.metrics)
            logging.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
            logging.info(f"--- Val Metrics ---")
            for key in scores.keys():
                logging.info("{}: {:.4f}".format(key, scores[key]))
            if args.monitor is not None:
                assert args.monitor in args.metrics, "monitor not in metrics!"
                score = scores[args.monitor]
                if is_best(best_score, score, args.monitor_criterion):
                    logging.info(
                        f"New best {args.monitor} score ({score:.4f}) "
                        f"at epoch-{epoch}, step-{global_step}"
                    )
                    best_dev_epoch = epoch
                    best_score = score
                    if best_checkpoint_path_name is not None:
                        state_dict = model.state_dict()
                        torch.save(state_dict, best_checkpoint_path_name)

        if epoch > args.unfreeze_epoch and epoch - best_dev_epoch >= args.max_epochs_before_stop:
            break

    logging.info('Best eval score: {:.4f} (at epoch {})'.format(best_score, best_dev_epoch))
    
    if os.path.isfile(best_checkpoint_path_name):
        logging.info("Loaded best model")
        state_dict = torch.load(best_checkpoint_path_name, map_location=args.cuda_choice)
        model.load_state_dict(state_dict)
    if test_dataloader is not None:
        scores, patient_embed_all, y_true_all = evaluate(model, test_dataloader, args.metrics)
        logging.info(f"--- Test ---")
        for key in scores.keys():
            logging.info("{}: {:.4f}".format(key, scores[key]))

if __name__ == "__main__":
    args = parse_args()
    main(args)
