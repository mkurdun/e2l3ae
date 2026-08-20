import argparse
import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import torch

from config import config
from models import L3AESymmetricGDModel
from utils import TimeBasedEvaluation


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--device", default=None, type=str)
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--validation", default="false", type=str)
parser.add_argument("--model", required=True, type=str, help="Path to saved e2l3ae model folder")
parser.add_argument("--k", default=100, type=int)
parser.add_argument("--output_csv", default=None, type=str)

args = parser.parse_args([] if "__file__" not in globals() else None)

if args.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_bool(x: str) -> bool:
    return str(x).strip().lower() == "true"


def load_evaluator(local_args):
    what = "val" if _to_bool(local_args.validation) else "test"
    if local_args.dataset not in config:
        raise ValueError(f"Unknown dataset: {local_args.dataset}")
    dataset, params = config[local_args.dataset]
    dataset.load_interactions(**params)
    evaluator = TimeBasedEvaluation(dataset, what=what)
    return evaluator


def main(local_args):
    torch.manual_seed(local_args.seed)
    np.random.seed(local_args.seed)

    evaluator = load_evaluator(local_args)
    model = L3AESymmetricGDModel.load_hybrid_infer_model(local_args.model, DEVICE)
    model.to(DEVICE)

    preds = model.predict_df(
        evaluator.test_src,
        texts=None,
        k=int(local_args.k),
        block_reminder=True,
    )
    results = evaluator(preds)
    row = {"model": local_args.model, "mode": "hybrid", **results}
    print(row)

    out_df = pd.DataFrame([row])
    if local_args.output_csv is None:
        split = "val" if _to_bool(local_args.validation) else "test"
        output_csv = os.path.join("results", f"e2l3ae_evaluate_{local_args.dataset}_{split}.csv")
    else:
        output_csv = local_args.output_csv
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved metrics to: {output_csv}")


if __name__ == "__main__":
    main(args)
