import argparse
import ast
import os
from copy import deepcopy

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config import config
from models import _build_eval_model_from_sbert
from utils import TimeBasedEvaluation
from layers import LayerSBERT


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--devices", default=None, type=str)
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--validation", default="false", type=str)
parser.add_argument("--model", required=True, type=str, help="Path to saved e2l3ae model folder")
parser.add_argument("--k", default=100, type=int)
parser.add_argument("--output_csv", default=None, type=str)
parser.add_argument("--sbert_batch_size", default=400, type=int)
parser.add_argument("--evaluate_coldstart", default="true", type=str)

args = parser.parse_args([] if "__file__" not in globals() else None)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_bool(x: str) -> bool:
    return str(x).strip().lower() == "true"


def _parse_devices(devices_raw: str | None) -> list[int] | None:
    if devices_raw is None:
        return None
    if not torch.cuda.is_available():
        raise ValueError("--devices is set but CUDA is not available.")
    try:
        device_ids = ast.literal_eval(devices_raw)
    except Exception as exc:
        raise ValueError("--devices must be a Python list, e.g. \"[0,1]\".") from exc
    if not isinstance(device_ids, list) or len(device_ids) == 0:
        raise ValueError("--devices must be a non-empty list, e.g. \"[0,1]\".")
    return [int(x) for x in device_ids]


def load_dataset_and_evaluator(local_args):
    what = "val" if _to_bool(local_args.validation) else "test"
    if local_args.dataset not in config:
        raise ValueError(f"Unknown dataset: {local_args.dataset}")
    dataset, params = config[local_args.dataset]
    dataset.load_interactions(**params)
    evaluator = TimeBasedEvaluation(dataset, what=what)
    train_interactions = (
        dataset.train_interactions
        if _to_bool(local_args.validation)
        else dataset.full_train_interactions
    )
    return dataset, evaluator, train_interactions


def collect_all_item_texts(local_args, dataset):
    items_d = dataset.items_texts.copy()
    items_d["asin"] = items_d.item_id

    am_itemids = items_d.asin.to_numpy()
    all_item_ids = np.array(dataset.all_interactions.item_id.cat.categories)
    all_df = pd.Series(all_item_ids).to_frame()
    all_df.columns = ["item_id"]
    am_df = pd.Series(am_itemids).to_frame().reset_index()
    am_df.columns = ["idx", "item_id"]
    locator = pd.merge(how="inner", left=all_df, right=am_df).idx.to_numpy()

    if local_args.dataset in config.keys():
        texts = items_d._text_attributes.to_numpy()[locator]
    else:
        texts = items_d.fillna(0).apply(
            lambda row: f"{row.title}: {'. '.join(eval(row.description))}",
            axis=1,
        ).to_numpy()[locator]
    return texts, pd.Index(all_item_ids)


def build_coldstart_evaluator(evaluator, train_interactions):
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
    cold_items_evaluator.test_target["user_id"] = (
        cold_items_evaluator.test_target.user_id.cat.remove_unused_categories()
    )
    cold_items_evaluator.test_src["user_id"] = (
        cold_items_evaluator.test_src.user_id.cat.remove_unused_categories()
    )
    cold_items_evaluator.test_target["item_id"] = (
        cold_items_evaluator.test_target.item_id.cat.remove_unused_categories()
    )
    candidates_df = pd.DataFrame(
        {
            "item_id": cold_items_evaluator.test_target.item_id.unique().to_numpy(),
            "user_id": "0",
            "value": 1.0,
        }
    )
    candidates_df["item_id"] = candidates_df["item_id"].astype("category")
    candidates_df["user_id"] = candidates_df["user_id"].astype("category")
    candidates_df["item_id"] = candidates_df["item_id"].cat.remove_unused_categories()
    cold_items_evaluator.candidates_df = candidates_df
    return cold_items_evaluator


def main(local_args):
    torch.manual_seed(local_args.seed)
    np.random.seed(local_args.seed)

    dataset, evaluator, train_interactions = load_dataset_and_evaluator(local_args)
    texts, all_item_ids = collect_all_item_texts(local_args, dataset)

    devices = _parse_devices(local_args.devices)
    model_device = DEVICE
    if devices is not None:
        model_device = torch.device(f"cuda:{devices[0]}")

    sbert_path = os.path.join(local_args.model, "sbert")
    sbert = SentenceTransformer(sbert_path, device=str(model_device)).to(model_device)

    sample_texts = list(texts[: min(2, len(texts))]) if len(texts) > 0 else [""]
    tokenized_sample = sbert.tokenize(sample_texts)
    sbert_for_eval_base = sbert

    if devices is not None and len(devices) > 1:
        sbert_for_eval_base = torch.nn.DataParallel(
            sbert,
            device_ids=devices,
            output_device=devices[0],
        )
    
    sbert_for_eval = LayerSBERT(
        model=sbert_for_eval_base,
        device=model_device,
        tokenized_sentences=tokenized_sample,
    )

    eval_model = _build_eval_model_from_sbert(
        sbert_model=sbert_for_eval,
        texts=texts,
        items_idx=all_item_ids,
        sbert_batch_size=int(local_args.sbert_batch_size),
        device=model_device,
    )
    preds = eval_model.predict_df(
        evaluator.test_src,
        k=int(local_args.k),
        block_reminder=True,
    )
    results = evaluator(preds)
    if _to_bool(local_args.evaluate_coldstart):
        coldstart_evaluator = build_coldstart_evaluator(evaluator, train_interactions)
        cold_candidates_df = getattr(coldstart_evaluator, "candidates_df", None)
        cold_preds = eval_model.predict_df(
            coldstart_evaluator.test_src,
            k=int(local_args.k),
            candidates_df=cold_candidates_df,
            block_reminder=True,
        )
        cold_results = coldstart_evaluator(cold_preds)
        cold_results = {("cold_start_" + k): v for k, v in cold_results.items()}
        results.update(cold_results)
    row = {"model": local_args.model, "mode": "encoder", **results}
    print(row)

    out_df = pd.DataFrame([row])
    if local_args.output_csv is None:
        split = "val" if _to_bool(local_args.validation) else "test"
        output_csv = os.path.join(
            "results",
            f"e2l3ae_evaluate_encoder_{local_args.dataset}_{split}.csv",
        )
    else:
        output_csv = local_args.output_csv
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved metrics to: {output_csv}")


if __name__ == "__main__":
    main(args)
