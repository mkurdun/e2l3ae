import argparse
import os
import subprocess
import time
from copy import deepcopy

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from callbacks import L3AEFinalizeBOnTrainEnd, evaluateWriter
from config import config
from dataloaders import InteractionBatchDataset
from models import L3AESymmetricGDModel
from schedules import LinearWarmup
from utils import NMSE, TimeBasedEvaluation, get_sparse_matrix_from_dataframe, preproces_html


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--device", default=None, type=str)
parser.add_argument("--devices", default=None, type=str)
parser.add_argument("--flag", default="none", type=str)
parser.add_argument("--validation", default="false", type=str)

parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--scheduler", default="none", type=str)
parser.add_argument("--init_lr", default=0.0, type=float)
parser.add_argument("--warmup_lr", default=1e-4, type=float)
parser.add_argument("--target_lr", default=1e-6, type=float)
parser.add_argument("--warmup_epochs", default=1, type=int)
parser.add_argument("--decay_epochs", default=3, type=int)
parser.add_argument("--tuning_epochs", default=1, type=int)
parser.add_argument("--epochs", default=5, type=int)

parser.add_argument("--dataset", default="-", type=str)
parser.add_argument("--prefix", default=None, type=str)

parser.add_argument("--sbert", default=None, type=str)
parser.add_argument("--max_seq_length", default=None, type=int)
parser.add_argument("--preproces_html", default="false", type=str)

parser.add_argument("--max_output", default=None, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--top_k", default=0, type=int)
parser.add_argument("--sbert_batch_size", default=200, type=int)

parser.add_argument("--l3ae_w_nmse", default=1.0, type=float)
parser.add_argument("--l3ae_w_align", default=1.0, type=float)
parser.add_argument("--l3ae_update_every", default=1, type=int)
parser.add_argument("--l3ae_cache_update_every", default=1, type=int)
parser.add_argument("--l3ae_warmup_steps", default=None, type=int)
parser.add_argument("--l3ae_residual_sum_gamma_s", default=1.0, type=float)

parser.add_argument("--lambda_s", default=10.0, type=float)
parser.add_argument("--lambda_b", default=100.0, type=float)
parser.add_argument("--lambda_r", default=150.0, type=float)

parser.add_argument("--model_name", default="my_model", type=str)

parser.add_argument("--evaluate", default="false", type=str)
parser.add_argument("--evaluate_epoch", default="false", type=str)
parser.add_argument("--save_every_epoch", default="true", type=str)
parser.add_argument("--eval_every_n_steps", default=0, type=int)
parser.add_argument("--save_every_n_steps", default=0, type=int)

args = parser.parse_args([] if "__file__" not in globals() else None)

if args.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

torch.set_float32_matmul_precision("medium")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_sbert(model_id: str, device: str, max_seq_length: int | None = None) -> SentenceTransformer:
    sbert = SentenceTransformer(model_id, device=device)
    if max_seq_length is not None:
        sbert.max_seq_length = int(max_seq_length)
    return sbert.to(device)


def load_data(local_args):
    what = "val" if local_args.validation == "true" else "test"
    if local_args.dataset not in config:
        print("Unknown dataset. Available datasets:")
        for dataset_name in config.keys():
            print(dataset_name)
        return None, None, None, None

    dataset, params = config[local_args.dataset]
    dataset.load_interactions(**params)
    evaluator = TimeBasedEvaluation(dataset, what=what)

    items_d = dataset.items_texts
    items_d["asin"] = items_d.item_id
    train_interactions = (
        dataset.train_interactions
        if local_args.validation == "true"
        else dataset.full_train_interactions
    )
    return dataset, evaluator, train_interactions, items_d


def load_text_model(local_args, items_d, dataset, train_interactions):
    if local_args.evaluate == "true" or local_args.evaluate_epoch == "true":
        am_itemids = items_d.asin.to_numpy()
        cc = np.array(dataset.all_interactions.item_id.cat.categories)
        ccdf = pd.Series(cc).to_frame()
        ccdf.columns = ["item_id"]
        amdf = pd.Series(am_itemids).to_frame().reset_index()
        amdf.columns = ["idx", "item_id"]
        am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()

        if local_args.dataset in config.keys():
            am_texts = items_d._text_attributes
        elif local_args.preproces_html == "true":
            am_texts = items_d.fillna(0).apply(
                lambda row: f"{row.title}: {preproces_html('. '.join(eval(row.description)))}",
                axis=1,
            )
        else:
            am_texts = items_d.fillna(0).apply(
                lambda row: f"{row.title}: {'. '.join(eval(row.description))}",
                axis=1,
            )
        am_texts_all = am_texts.to_numpy()[am_locator]
    else:
        am_texts_all = None

    am_itemids = items_d.asin.to_numpy()
    cc = np.array(train_interactions.item_id.cat.categories)
    ccdf = pd.Series(cc).to_frame()
    ccdf.columns = ["item_id"]
    amdf = pd.Series(am_itemids).to_frame().reset_index()
    amdf.columns = ["idx", "item_id"]
    am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()
    am_texts = items_d._text_attributes.to_numpy()[am_locator]

    if local_args.prefix is not None:
        am_texts = np.array([local_args.prefix + text for text in am_texts])

    sbert = build_sbert(
        local_args.sbert,
        DEVICE,
        max_seq_length=local_args.max_seq_length,
    )
    am_tokenized = sbert.tokenize(am_texts)
    if am_texts_all is None:
        am_texts_all = am_texts
    return am_texts_all, am_tokenized, sbert


def prepare_schedule(local_args, steps_per_epoch):
    if local_args.scheduler == "CosineDecay":
        schedule = keras.optimizers.schedules.CosineDecay(
            0.0,
            steps_per_epoch * (local_args.decay_epochs + local_args.warmup_epochs),
            alpha=0.0,
            name="CosineDecay",
            warmup_target=local_args.warmup_lr,
            warmup_steps=steps_per_epoch * local_args.warmup_epochs,
        )
        epochs = local_args.warmup_epochs + local_args.decay_epochs + local_args.tuning_epochs
    elif local_args.scheduler == "LinearWarmup":
        schedule = LinearWarmup(
            warmup_steps=steps_per_epoch * local_args.warmup_epochs,
            decay_steps=steps_per_epoch * local_args.decay_epochs,
            starting_lr=local_args.init_lr,
            warmup_lr=local_args.warmup_lr,
            final_lr=local_args.target_lr,
        )
        epochs = local_args.warmup_epochs + local_args.decay_epochs + local_args.tuning_epochs
    else:
        schedule = local_args.lr
        epochs = local_args.epochs
    return schedule, epochs


def main(local_args):
    folder = os.path.join(
        "results",
        f"{str(pd.Timestamp('today'))} {9 * int(1e6) + np.random.randint(999999)}".replace(" ", "_"),
    )
    os.makedirs(folder, exist_ok=True)

    run_setup = vars(local_args).copy()
    run_setup["cuda_or_cpu"] = str(DEVICE)
    pd.Series(run_setup).to_csv(f"{folder}/setup.csv")

    torch.manual_seed(local_args.seed)
    keras.utils.set_random_seed(local_args.seed)
    np.random.seed(local_args.seed)

    dataset, evaluator, train_interactions, items_d = load_data(local_args)
    if dataset is None:
        return

    am_texts_all, am_tokenized, sbert = load_text_model(
        local_args, items_d, dataset, train_interactions
    )

    if local_args.devices is not None:
        devices_to_run = eval(local_args.devices)
        module_sbert = torch.nn.DataParallel(
            sbert, device_ids=devices_to_run, output_device=devices_to_run[0]
        )
    else:
        module_sbert = sbert

    X = get_sparse_matrix_from_dataframe(train_interactions)
    datal = InteractionBatchDataset(
        X,
        am_tokenized,
        DEVICE,
        shuffle=True,
        max_output=local_args.max_output,
        batch_size=local_args.batch_size,
        include_full_user_row=False,
    )
    steps_per_epoch = len(datal)
    print(sbert)

    model = L3AESymmetricGDModel(
        tokenized_sentences=am_tokenized,
        items_idx=train_interactions.item_id.cat.categories,
        sbert=module_sbert,
        device=DEVICE,
        top_k=local_args.top_k,
        sbert_batch_size=local_args.sbert_batch_size,
        X_train=X,
        lambda_s=float(local_args.lambda_s),
        lambda_b=float(local_args.lambda_b),
        lambda_r=float(local_args.lambda_r),
        l3ae_w_nmse=float(local_args.l3ae_w_nmse),
        l3ae_w_align=float(local_args.l3ae_w_align),
        update_every=int(local_args.l3ae_update_every),
        cache_update_every=int(local_args.l3ae_cache_update_every),
        warmup_steps=(
            int(local_args.l3ae_warmup_steps) if local_args.l3ae_warmup_steps is not None else None
        ),
        l3ae_residual_sum_gamma_s=float(local_args.l3ae_residual_sum_gamma_s),
    )

    schedule, epochs = prepare_schedule(local_args, steps_per_epoch)
    model.to(DEVICE)

    callbacks = [L3AEFinalizeBOnTrainEnd()]
    eval_cb = None
    if (
        local_args.evaluate == "true"
        or local_args.evaluate_epoch == "true"
        or local_args.save_every_epoch == "true"
        or local_args.eval_every_n_steps > 0
    ):
        coldstart_evaluator = None
        train_items = train_interactions[["user_id", "item_id"]].drop_duplicates()["item_id"].unique()
        test_items = evaluator.test_target.item_id.unique().tolist()
        cold_items = list(set(test_items) - set(train_items))
        cold_items_evaluator = deepcopy(evaluator)
        cold_items_evaluator.test_target = cold_items_evaluator.test_target[
            cold_items_evaluator.test_target.item_id.isin(cold_items)
        ]
        cold_items_evaluator.test_src = cold_items_evaluator.test_src[
            cold_items_evaluator.test_src.user_id.isin(cold_items_evaluator.test_target.user_id)
        ]
        cold_items_evaluator.test_target["user_id"] = cold_items_evaluator.test_target.user_id.cat.remove_unused_categories()
        cold_items_evaluator.test_src["user_id"] = cold_items_evaluator.test_src.user_id.cat.remove_unused_categories()
        coldstart_evaluator = cold_items_evaluator
        
        eval_cb = evaluateWriter(
            items_idx=train_interactions.item_id.cat.categories,
            all_items_idx=dataset.all_interactions.item_id.cat.categories,
            sbert=sbert,
            evaluator=evaluator,
            logdir=folder,
            DEVICE=DEVICE,
            texts=am_texts_all,
            sbert_name=local_args.model_name,
            evaluate_epoch=local_args.evaluate_epoch,
            save_every_epoch=local_args.save_every_epoch,
            coldstart_evaluator=coldstart_evaluator,
            eval_model=model,
            eval_every_n_steps=int(local_args.eval_every_n_steps),
            save_every_n_steps=int(local_args.save_every_n_steps),
        )
        callbacks.append(eval_cb)

    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=schedule),
        loss=NMSE,
        metrics=[keras.metrics.CosineSimilarity()],
    )
    model.train_step(datal[0])
    model._l3ae_train_step = 0
    model.built = True
    model.summary()
    print("Starting training loop")

    train_time_start = time.time()
    fit_result = model.fit(
        datal,
        initial_epoch=0,
        epochs=epochs,
        callbacks=callbacks,
    )
    train_time = time.time() - train_time_start

    model.save(local_args.model_name)

    if local_args.evaluate == "true" and eval_cb is not None:
        eval_cb.on_epoch_end(epoch="final")


if __name__ == "__main__":
    main(args)
