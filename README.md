# E²L³AE

This is implementation of **E²L³AE: End‑to‑End Text Encoder Tuning for LLM‑Enhanced Linear Autoencoders**.

L³AE integrates frozen LLM embeddings into linear autoencoders, but fixed text representations can be suboptimal for recommendation because consumption‑relevant item characteristics often deviate from general textual semantics. We propose an end‑to‑end extension that fine‑tunes the text encoder within the same linear autoencoder framework, directly shaping its item embeddings using item closeness derived from user–item interactions. This integrates collaborative information into the encoder, adapting pretrained textual knowledge to recommendation objectives while preserving semantic richness. Experiments on public datasets show that our end-to-end trained hybrid model consistently outperforms the original L³AE and other baselines. Moreover, the fine‑tuned encoder alone serves as a strong semantic‑based recommender. We release our code to support future research.

## Training Example

Install dependencies first:

```bash
cd e2l3ae
pip install -r requirements.txt
```

```bash
python train.py \
  --seed 42 \
  --scheduler None \
  --lr 1e-4 \
  --epochs 100 \
  --dataset beauty \
  --devices "[0,1,2,3,4,5,6,7]" \
  --validation false \
  --max_seq_length 384 \
  --max_output 10000 \
  --batch_size 1024 \
  --sbert_batch_size 1024 \
  --evaluate true \
  --evaluate_epoch false \
  --save_every_epoch false \
  --model_name my_model \
  --sbert "all-mpnet-base-v2" \
  --l3ae_w_nmse 250.0 \
  --l3ae_w_align 1.0 \
  --lambda_s 0.5 \
  --lambda_b 500.0 \
  --lambda_r 1.0 \
  --l3ae_update_every 292 \
  --eval_every_n_steps 292 \
  --save_every_n_steps 292 \
  --l3ae_residual_sum_gamma_s 1.0
```

## Evaluate Hybrid Model

Evaluates the saved hybrid matrix:

```bash
python evaluate.py \
  --seed 42 \
  --dataset beauty \
  --validation false \
  --model my_model \
  --k 100
```

## Evaluate Encoder-Only Model

Evaluates only the encoder, separately from the hybrid matrix:

```bash
python evaluate_encoder.py \
  --seed 42 \
  --dataset beauty \
  --validation false \
  --model my_model \
  --devices "[0,1,2,3,4,5,6,7]" \
  --k 100 \
  --evaluate_coldstart true
```

## Ready-to-Use Launch Scripts (all-mpnet-base-v2)

Scripts are located in `runs`:

- `train_beauty.sh`
- `train_toys.sh`
- `train_grocery.sh`
- `train_office.sh`

