import json
import logging
import math
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from tqdm import tqdm

from dataloaders import PredictDfRecSysDataset
from layers import LayerSBERT
from utils import get_first_item, get_sparse_matrix_from_dataframe

logger = logging.getLogger(__name__)


def clip_gradients(trainable_weights, max_norm: float):
    for w in trainable_weights:
        g = w.value.grad
        if g is None:
            continue
        g_norm = torch.linalg.vector_norm(g.detach()).item()
        if g_norm > max_norm:
            g.mul_(max_norm / g_norm)


def _compute_B_residual_ease_closed_form(
    X_train: sp.csr_matrix,
    M: np.ndarray,
    lambda_b: float,
) -> np.ndarray:
    X = X_train.tocsr()
    M = np.asarray(M, dtype=np.float64)

    XtX = (X.T @ X).toarray().astype(np.float64)
    A = XtX + float(lambda_b) * np.eye(XtX.shape[0], dtype=np.float64)
    P = np.linalg.inv(A)

    rhs = XtX @ (np.eye(XtX.shape[0], dtype=np.float64) - M)
    U = P @ rhs
    mu = np.diag(U) / np.diag(P)
    B = U - P * mu[np.newaxis, :]
    np.fill_diagonal(B, 0.0)
    return B.astype(np.float32)


def _encode_texts_forward(sbert_model, texts, sbert_batch_size=400, device="cuda"):
    text_list = list(texts)
    batches = []
    with torch.no_grad():
        for start in range(0, len(text_list), sbert_batch_size):
            end = min(start + sbert_batch_size, len(text_list))
            tokenized = sbert_model.tokenize(text_list[start:end])
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            batch_embs = sbert_model(tokenized, training=False)
            batches.append(batch_embs)
    return torch.cat(batches, dim=0)


def _build_eval_model_from_sbert(
    sbert_model,
    texts,
    items_idx=None,
    sbert_batch_size=400,
    device="cuda",
):
    embs = _encode_texts_forward(
        sbert_model=sbert_model,
        texts=texts,
        sbert_batch_size=sbert_batch_size,
        device=device,
    )
    emb_dim = embs.shape[1]
    model = SparseKerasELSA(len(items_idx), emb_dim, items_idx, device=device)
    model.set_weights([embs])
    model.to(device)
    return model


class SparseKerasEASE(keras.models.Model):
    def __init__(self, items_idx, device, B: torch.Tensor | None = None):
        super().__init__()
        self.items_idx = items_idx
        self.device = device
        self._B = B

    def call(self, x, training=None):
        return x @ self._B.to(x.device)

    def predict_on_batch(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
            y = self.call(x_t, training=False)
        return y.cpu().numpy().astype(np.float32)

    def save(self, path: str, **kwargs) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "B.npy"), self._B.numpy())
        np.save(os.path.join(path, "items_idx.npy"), np.array(self.items_idx))

    @classmethod
    def load(cls, path: str, device) -> "SparseKerasEASE":
        items_idx = np.load(os.path.join(path, "items_idx.npy"), allow_pickle=True)
        B = torch.from_numpy(np.load(os.path.join(path, "B.npy")))
        return cls(items_idx, device, B=B)

    def predict_df(
        self,
        df,
        texts,
        items_idx=None,
        k=100,
        user_ids=None,
        candidates_df=None,
        block_reminder=True,
    ):
        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        if candidates_df is not None:
            candidates_vec = get_sparse_matrix_from_dataframe(
                candidates_df, item_indices=self.items_idx
            ).toarray()
            candidates_vec = torch.from_numpy(candidates_vec)

        data = PredictDfRecSysDataset(df, self.items_idx, batch_size=16384)
        dfs = []

        for i in tqdm(range(len(data)), total=len(data)):
            x, batch_uids = data[i]
            batch = torch.from_numpy(self.predict_on_batch(x))

            if block_reminder:
                batch = batch * (1 - x.astype(bool))
            if candidates_df is not None:
                batch *= candidates_vec

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            chunk = pd.DataFrame(
                {
                    "user_id": np.stack([batch_uids] * k).flatten("F"),
                    "item_id": np.array(self.items_idx)[indices_].flatten(),
                    "value": values_.flatten(),
                }
            )
            chunk["user_id"] = chunk["user_id"].astype(str).astype("category")
            chunk["item_id"] = chunk["item_id"].astype(str).astype("category")
            dfs.append(chunk)

        out = pd.concat(dfs)
        out["user_id"] = out["user_id"].astype(str).astype("category")
        out["item_id"] = out["item_id"].astype(str).astype("category")
        return out


class SparseKerasELSA(keras.models.Model):
    def __init__(self, n_items, n_dims, items_idx, device, A=None):
        super().__init__()
        self.device = device
        self.items_idx = items_idx
        self._A = None
        if A is not None:
            self.set_weights([A])

    def set_weights(self, weights):
        if not weights:
            raise ValueError("SparseKerasELSA.set_weights expects embeddings tensor.")
        A = weights[0]
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A)
        A = A.to(self.device, dtype=torch.float32)
        self._A = torch.nn.functional.normalize(A, dim=-1)

    def predict_on_batch(self, x: np.ndarray) -> np.ndarray:
        if self._A is None:
            raise ValueError("SparseKerasELSA.predict_on_batch: embeddings are not initialized.")
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
            xA = x_t @ self._A
            xAAT = xA @ self._A.T
            y = keras.activations.relu(xAAT - x_t)
        return y.cpu().numpy().astype(np.float32)

    def predict_df(
        self,
        df,
        texts=None,
        items_idx=None,
        k=100,
        user_ids=None,
        candidates_df=None,
        block_reminder=True,
    ):
        if user_ids is None:
            user_ids = np.array(df.user_id.cat.categories)

        if candidates_df is not None:
            candidates_vec = get_sparse_matrix_from_dataframe(
                candidates_df, item_indices=self.items_idx
            ).toarray()
            candidates_vec = torch.from_numpy(candidates_vec)

        data = PredictDfRecSysDataset(df, self.items_idx, batch_size=16384)
        dfs = []
        for i in tqdm(range(len(data)), total=len(data)):
            x, batch_uids = data[i]
            batch = torch.from_numpy(self.predict_on_batch(x))

            if block_reminder:
                batch = batch * (1 - x.astype(bool))
            if candidates_df is not None:
                batch *= candidates_vec

            values_, indices_ = torch.topk(batch.to("cpu"), k)
            chunk = pd.DataFrame(
                {
                    "user_id": np.stack([batch_uids] * k).flatten("F"),
                    "item_id": np.array(self.items_idx)[indices_].flatten(),
                    "value": values_.flatten(),
                }
            )
            chunk["user_id"] = chunk["user_id"].astype(str).astype("category")
            chunk["item_id"] = chunk["item_id"].astype(str).astype("category")
            dfs.append(chunk)

        out = pd.concat(dfs)
        out["user_id"] = out["user_id"].astype(str).astype("category")
        out["item_id"] = out["item_id"].astype(str).astype("category")
        return out


class L3AE(keras.models.Model):
    def __init__(
        self,
        sbert=None,
        device=None,
        tokenized_sentences=None,
        items_idx=None,
        lambda_s: float = 1.0,
        lambda_b: float = 500.0,
        lambda_r: float = 10.0,
        sbert_batch_size: int = 128,
        layer_sbert=None,
        freeze_backbone: bool = True,
    ):
        if (sbert is None) == (layer_sbert is None):
            raise TypeError(
                "L3AE: pass exactly one of sbert or layer_sbert."
            )
        super().__init__()
        self.device = device
        self.items_idx = items_idx
        self.tokenized_sentences = tokenized_sentences
        self.sbert_batch_size = sbert_batch_size
        self.lambda_s = lambda_s
        self.lambda_b = lambda_b
        self.lambda_r = lambda_r

        if layer_sbert is not None:
            self.sbert = layer_sbert
        else:
            self.sbert = LayerSBERT(sbert, device, tokenized_sentences)
        if freeze_backbone:
            for param in self.sbert.sbert.parameters():
                param.requires_grad_(False)

    def call(self, x, training=None):
        return self.sbert(x, training=training)

    def _encode_all_items(self) -> np.ndarray:
        tokenized_items = self.tokenized_sentences
        n_total = get_first_item(tokenized_items).shape[0]
        n_batches = math.ceil(n_total / self.sbert_batch_size)

        parts = []
        with torch.no_grad():
            for i in range(n_batches):
                lo = i * self.sbert_batch_size
                hi = lo + self.sbert_batch_size
                batch = {k: v[lo:hi].to(self.device) for k, v in tokenized_items.items()}
                parts.append(self.sbert(batch, training=False).cpu())

        return torch.nn.functional.normalize(torch.cat(parts, dim=0), dim=-1).numpy()

    @staticmethod
    def _compute_S(
        F: np.ndarray,
        lambda_s: float,
        B: np.ndarray | None = None,
        lambda_r: float = 0.0,
    ) -> np.ndarray:
        n = F.shape[0]
        F = F.astype(np.float64)
        G_Q = F @ F.T
        use_anchor = B is not None and float(lambda_r) > 0.0
        if use_anchor:
            P_Q = np.linalg.inv(
                G_Q + (float(lambda_s) + float(lambda_r)) * np.eye(n)
            )
            rhs = G_Q + float(lambda_r) * B.astype(np.float64)
        else:
            P_Q = np.linalg.inv(G_Q + float(lambda_s) * np.eye(n))
            rhs = G_Q
        M = P_Q @ rhs
        mu = np.diag(M) / np.diag(P_Q)
        S = M - P_Q * mu[np.newaxis, :]
        np.fill_diagonal(S, 0.0)
        return S.astype(np.float32)

    @staticmethod
    def _compute_B(
        X: sp.csr_matrix,
        S: np.ndarray,
        lambda_b: float,
        lambda_r: float,
    ) -> np.ndarray:
        n = X.shape[1]
        use_S = lambda_r > 0.0
        G_X = (X.T @ X).toarray().astype(np.float64)
        P = np.linalg.inv(G_X + (lambda_b + (lambda_r if use_S else 0.0)) * np.eye(n))
        if use_S:
            S = S.astype(np.float64)
            PS = P @ S
            mu = (1.0 + lambda_r * np.diag(PS)) / np.diag(P)
            B = np.eye(n) + lambda_r * PS - P * mu[np.newaxis, :]
        else:
            B = np.eye(n) - P / np.diag(P)[np.newaxis, :]
        np.fill_diagonal(B, 0.0)
        return B.astype(np.float32)


class L3AESymmetricGDModel(keras.models.Model):

    def __init__(
        self,
        tokenized_sentences,
        items_idx,
        sbert,
        device,
        X_train: sp.csr_matrix,
        top_k: int = 0,
        sbert_batch_size: int = 128,
        asym_params_lr_scaling: float = 1.0,
        lambda_s: float = 1.0,
        lambda_b: float = 500.0,
        lambda_r: float = 10.0,
        l3ae_w_nmse: float = 1.0,
        l3ae_w_align: float = 1.0,
        update_every: int = 1,
        cache_update_every: int = 1,
        warmup_steps: int | None = None,
        l3ae_residual_sum_gamma_s: float = 1.0,
    ):
        super().__init__()
        self.device = device
        self.items_idx = items_idx
        self.tokenized_sentences = tokenized_sentences
        self.top_k = int(top_k)
        self.sbert_batch_size = int(sbert_batch_size)
        self.asym_params_lr_scaling = float(asym_params_lr_scaling)

        self._X_train_csr = X_train
        self.lambda_s = float(lambda_s)
        self.lambda_b = float(lambda_b)
        self.lambda_r = float(lambda_r)
        self.l3ae_w_nmse = float(l3ae_w_nmse)
        self.l3ae_w_align = float(l3ae_w_align)

        ue = int(update_every)
        if ue < 1:
            raise ValueError("update_every must be >= 1")
        self.update_every = ue
        self.cache_update_every = int(cache_update_every)
        ws = int(warmup_steps) if warmup_steps is not None else ue
        self.l3ae_warmup_steps = ws

        self.l3ae_residual_sum_gamma_s = float(l3ae_residual_sum_gamma_s)

        self._S_global_np: np.ndarray | None = None
        self._B_global_np: np.ndarray | None = None
        self._F_cache_np: np.ndarray | None = None
        self._S_semantic_cf_cache_np: np.ndarray | None = None
        self._l3ae_train_step: int = 0
        self._l3ae_embedding_stale: bool = True
        self._l3ae_eval_W_np: np.ndarray | None = None
        self._l3ae_eval_stale: bool = True

        self._l3ae_shared_cf = L3AE(
            sbert=sbert,
            device=self.device,
            tokenized_sentences=tokenized_sentences,
            items_idx=items_idx,
            lambda_s=self.lambda_s,
            lambda_b=self.lambda_b,
            lambda_r=self.lambda_r,
            sbert_batch_size=self.sbert_batch_size,
            freeze_backbone=False,
        )

    def _l3ae_compute_b_cf(self, S_np: np.ndarray) -> np.ndarray:
        gamma_s = float(self.l3ae_residual_sum_gamma_s)
        return _compute_B_residual_ease_closed_form(
            self._X_train_csr,
            (gamma_s * S_np.astype(np.float64)).astype(np.float32),
            self.lambda_b,
        )

    def _l3ae_refresh_embedding_cache(self) -> None:
        F_np = self._l3ae_shared_cf._encode_all_items()
        self._F_cache_np = np.asarray(F_np, dtype=np.float32).copy()
        self._S_semantic_cf_cache_np = None
        self._l3ae_embedding_stale = False

    def _l3ae_get_all_item_embeddings_cached_np(self) -> np.ndarray:
        if self._F_cache_np is not None:
            return self._F_cache_np
        self._l3ae_refresh_embedding_cache()
        return self._F_cache_np

    def _l3ae_get_cached_f_tensor(
        self,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._F_cache_np is None:
            self._l3ae_refresh_embedding_cache()
        return torch.as_tensor(self._F_cache_np, device=self.device, dtype=dtype).detach()

    def _align_S_off_tensor(self, Q_neg: torch.Tensor, item_global_idx: torch.Tensor) -> torch.Tensor:
        if self._S_global_np is None:
            return None
        g = item_global_idx.detach().long().cpu().numpy()
        S_np = self._S_global_np[np.ix_(g, g)]
        S_t = torch.as_tensor(S_np, device=self.device, dtype=Q_neg.dtype).detach()
        return S_t - torch.diag_embed(torch.diag(S_t))

    def _l3ae_align_loss(
        self,
        A: torch.Tensor,
        item_global_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self._S_global_np is None:
            return torch.tensor(0.0, device=A.device, dtype=A.dtype)
        F_cache = self._l3ae_get_cached_f_tensor(A.dtype)

        g = item_global_idx.detach().long()
        g_np = g.cpu().numpy()
        S_cols = torch.as_tensor(self._S_global_np[:, g_np], device=self.device, dtype=A.dtype).detach()
        S_gg = S_cols.index_select(0, g)

        base_pred = F_cache.mT @ S_cols
        q_delta = A - F_cache.index_select(0, g)
        pred = base_pred + q_delta.mT @ S_gg
        return torch.mean((A.mT - pred) ** 2)

    def _l3ae_get_semantic_s_cf_cached(self) -> np.ndarray:
        if self._S_semantic_cf_cache_np is not None:
            return self._S_semantic_cf_cache_np
        S_sem_np = self._recompute_s_numpy(couple_b_anchor=False)
        self._S_semantic_cf_cache_np = np.asarray(S_sem_np, dtype=np.float32).copy()
        return self._S_semantic_cf_cache_np

    def _recompute_s_numpy_residual_anchor(self, S_semantic_np: np.ndarray | None = None) -> np.ndarray:
        if self._B_global_np is None:
            raise ValueError("Residual-sum S update requires initialized B.")
        X = self._X_train_csr
        n = int(X.shape[1])
        a = float(self.l3ae_residual_sum_gamma_s)
        w_a = float(self.l3ae_w_align)
        lam_s = float(self.lambda_s)
        lam_r = float(self.lambda_r)

        G_X = (X.T @ X).toarray().astype(np.float64)
        B = self._B_global_np.astype(np.float64)
        XtY = G_X @ (np.eye(n, dtype=np.float64) - B)
        if S_semantic_np is None:
            raise ValueError("Semantic anchor requires S_semantic_np.")
        S_sem = S_semantic_np.astype(np.float64)
        A = (lam_r * (a ** 2)) * G_X + (lam_s + w_a) * np.eye(n, dtype=np.float64)
        rhs = (lam_r * a) * XtY + w_a * S_sem
        P = np.linalg.inv(A)
        M = P @ rhs
        mu = np.diag(M) / np.diag(P)
        S = M - P * mu[np.newaxis, :]
        np.fill_diagonal(S, 0.0)
        return S.astype(np.float32)

    def _recompute_s_numpy(self, *, couple_b_anchor: bool = True) -> np.ndarray:
        w_a = float(self.l3ae_w_align)
        if w_a > 0.0:
            ls = self.lambda_s / w_a
            lr = self.lambda_r / w_a
        else:
            ls, lr = self.lambda_s, self.lambda_r

        B_np: np.ndarray | None = None
        if couple_b_anchor and float(self.lambda_r) > 0.0:
            B_np = self._B_global_np

        if B_np is not None and float(self.lambda_r) > 0.0:
            S_sem_np = self._l3ae_get_semantic_s_cf_cached()
            return self._recompute_s_numpy_residual_anchor(S_semantic_np=S_sem_np)

        F = self._l3ae_get_all_item_embeddings_cached_np()
        if B_np is not None:
            S_sem_np = L3AE._compute_S(F, ls, B_np, lr)
        else:
            S_sem_np = L3AE._compute_S(F, ls)
        if not couple_b_anchor:
            self._S_semantic_cf_cache_np = np.asarray(S_sem_np, dtype=np.float32).copy()
        return S_sem_np

    def l3ae_refresh_b(self) -> None:
        self._l3ae_ensure_synced_with_weights()

    def l3ae_on_epoch_end(self) -> None:
        self.l3ae_refresh_b()

    def _l3ae_ensure_embedding_fresh(self) -> None:
        if self._l3ae_embedding_stale or self._F_cache_np is None:
            self._l3ae_refresh_embedding_cache()

    def _l3ae_ensure_synced_with_weights(self) -> None:
        self._l3ae_ensure_embedding_fresh()
        if self._S_semantic_cf_cache_np is None or self._B_global_np is None:
            S = self._recompute_s_numpy(couple_b_anchor=False)
            self._B_global_np = self._l3ae_compute_b_cf(S)

    def _l3ae_do_closed_form_update(self, s: int) -> None:
        self._l3ae_ensure_embedding_fresh()
        S_for_b = self._recompute_s_numpy(couple_b_anchor=False)
        self._B_global_np = self._l3ae_compute_b_cf(S_for_b)
        self._S_global_np = self._recompute_s_numpy()

    def _l3ae_maybe_closed_form_tick(self, s: int) -> None:
        w = self.l3ae_warmup_steps
        if s < w or (s - w) % self.update_every != 0:
            return
        self._l3ae_do_closed_form_update(s)

    def get_l3ae_infer_weight_numpy(self) -> np.ndarray | None:
        a = float(self.l3ae_residual_sum_gamma_s)
        embedding_fresh = not self._l3ae_embedding_stale and self._F_cache_np is not None

        if embedding_fresh and self._S_semantic_cf_cache_np is not None and self._B_global_np is not None:
            W = self._B_global_np.astype(np.float64) + a * self._S_semantic_cf_cache_np.astype(np.float64)
            np.fill_diagonal(W, 0.0)
            return W.astype(np.float32)

        if self._l3ae_eval_stale or self._l3ae_eval_W_np is None:
            if embedding_fresh:
                F_np = self._F_cache_np
            else:
                F_np = np.asarray(self._l3ae_shared_cf._encode_all_items(), dtype=np.float32)
            w_a = float(self.l3ae_w_align)
            ls = self.lambda_s / w_a if w_a > 0.0 else self.lambda_s
            S_np = L3AE._compute_S(F_np, ls)
            B_np = self._l3ae_compute_b_cf(S_np)

            W = B_np.astype(np.float64) + a * S_np.astype(np.float64)
            np.fill_diagonal(W, 0.0)
            self._l3ae_eval_W_np = W.astype(np.float32)
            self._l3ae_eval_stale = False

        return self._l3ae_eval_W_np

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._l3ae_shared_cf.sbert.sb().save(os.path.join(path, "sbert"))

        cfg = {
            "class": "L3AESymmetricGDModel",
            "top_k": int(self.top_k),
            "sbert_batch_size": int(self.sbert_batch_size),
            "lambda_s": float(self.lambda_s),
            "lambda_b": float(self.lambda_b),
            "lambda_r": float(self.lambda_r),
            "l3ae_w_nmse": float(self.l3ae_w_nmse),
            "l3ae_w_align": float(self.l3ae_w_align),
            "update_every": int(self.update_every),
            "warmup_steps": int(self.l3ae_warmup_steps),
            "l3ae_residual_sum_gamma_s": float(self.l3ae_residual_sum_gamma_s),
            "_l3ae_train_step": int(self._l3ae_train_step),
        }
        with open(os.path.join(path, "l3ae_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        W_np = self.get_l3ae_infer_weight_numpy()
        np.save(os.path.join(path, "W.npy"), np.asarray(W_np, dtype=np.float32))
        np.save(os.path.join(path, "items_idx.npy"), np.array(self.items_idx))

    @staticmethod
    def load_hybrid_infer_model(path: str, device) -> SparseKerasEASE:
        items_idx = pd.Index(np.load(os.path.join(path, "items_idx.npy"), allow_pickle=True))
        W = torch.from_numpy(np.load(os.path.join(path, "W.npy"))).to(device)
        return SparseKerasEASE(items_idx, device, B=W)

    def predict_df(
        self,
        df,
        texts,
        items_idx=None,
        k=100,
        user_ids=None,
        candidates_df=None,
        block_reminder=True,
    ):
        W_np = self.get_l3ae_infer_weight_numpy()
        if W_np is None:
            raise ValueError("L3AESymmetricGDModel.predict_df: B is not initialized.")
        W_t = torch.from_numpy(np.asarray(W_np, dtype=np.float32))
        ease = SparseKerasEASE(self.items_idx, self.device, B=W_t)
        ease.to(self.device)
        return ease.predict_df(
            df,
            k=k,
            texts=texts,
            items_idx=items_idx,
            user_ids=user_ids,
            candidates_df=candidates_df,
            block_reminder=block_reminder,
        )

    def predict_df_encoder(
        self,
        df,
        texts,
        items_idx=None,
        k=100,
        user_ids=None,
        candidates_df=None,
        block_reminder=True,
    ):
        sbert_model = self._l3ae_shared_cf.sbert
        eval_model = _build_eval_model_from_sbert(
            sbert_model=sbert_model,
            texts=texts,
            items_idx=items_idx,
            sbert_batch_size=self.sbert_batch_size,
            device=self.device,
        )
        return eval_model.predict_df(
            df,
            k=k,
            texts=texts,
            items_idx=items_idx,
            user_ids=user_ids,
            candidates_df=candidates_df,
            block_reminder=block_reminder,
        )

    
    def train_step(self, data):
        if self._l3ae_train_step == 0 and self.l3ae_warmup_steps == 0:
            self._l3ae_do_closed_form_update(0)
        self._l3ae_train_step += 1
        s = self._l3ae_train_step

        a, b = data
        x, y = a
        y = torch.hstack((x, y))
        x_out = y
        (
            tokenized_items,
            slicer,
            negative_slicer,
            item_global_idx,
            user_row_idx,
        ) = b
        slicer = slicer.to(self.device)
        negative_slicer = negative_slicer.to(self.device)
        item_global_idx = item_global_idx.to(self.device)
        user_row_idx = user_row_idx.to(self.device)

        self.zero_grad()

        sbert_batch_size = self.sbert_batch_size
        len_sentences = get_first_item(tokenized_items).shape[0]
        max_i = math.ceil(len_sentences / sbert_batch_size)

        cpu_rng_state = torch.clone(torch.random.get_rng_state())
        cuda_rng_states_raw = torch.cuda.get_rng_state_all() 
        cuda_rng_states = []        
        for orig_state in cuda_rng_states_raw:  
            cuda_rng_states.append(orig_state.clone())

        with torch.no_grad():
            A_batches = []
            for i in range(max_i):
                ind_min = i * sbert_batch_size
                ind_max = ind_min + sbert_batch_size
                a_b = self._l3ae_shared_cf(
                    {k: v[ind_min:ind_max] for k, v in tokenized_items.items()},
                    training=True,
                )
                A_batches.append(a_b)
            A_all = torch.cat(A_batches, dim=0)

        A_all.requires_grad = True
        A_slicer = torch.nn.functional.normalize(A_all[slicer], dim=-1)
        A_neg = torch.nn.functional.normalize(A_all[negative_slicer], dim=-1)

        xA = torch.matmul(x, A_slicer)
        xAAT = torch.matmul(xA, A_neg.T)
        raw_scores = xAAT - x_out
        y_pred = raw_scores + (keras.activations.relu(raw_scores) - raw_scores).detach()

        if self.top_k > 0:
            val, inds = torch.topk(y_pred, self.top_k)
            y = torch.gather(y, 1, inds)
            y_pred = val

        l_nmse = self.compute_loss(y=y, y_pred=y_pred)
        l_align = self._l3ae_align_loss(A_neg, item_global_idx)

        corrected_weight = self.l3ae_w_align / self.l3ae_w_nmse
        loss = l_nmse + corrected_weight * l_align
        frac_pos_raw = float((raw_scores > 0).float().mean().detach().item())
        loss.backward()

        grad_norm_a = float(torch.linalg.vector_norm(A_all.grad.detach()).item())
        if grad_norm_a > 5.0:
            A_all.grad = A_all.grad / grad_norm_a * 5.0
        if grad_norm_a < 1e-7:
            A_all.grad = A_all.grad * (1e-7 / grad_norm_a)

        torch.random.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_states)

        for i in range(max_i):
            ind_min = i * sbert_batch_size
            ind_max = ind_min + sbert_batch_size
            t_out = self._l3ae_shared_cf(
                {k: v[ind_min:ind_max] for k, v in tokenized_items.items()},
                training=True,
            )
            t_out.backward(A_all.grad[ind_min:ind_max])

        trainable_weights = [v for v in self._l3ae_shared_cf.trainable_weights]
        clip_gradients(trainable_weights, 5.0)
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        self._l3ae_embedding_stale = True
        self._l3ae_eval_stale = True

        self._l3ae_maybe_closed_form_tick(s)
        if self._l3ae_train_step % self.cache_update_every == 0:
            self._l3ae_ensure_embedding_fresh()

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        out = {m.name: m.result() for m in self.metrics}
        out["l3ae_step"] = s
        return out
