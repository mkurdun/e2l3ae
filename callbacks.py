import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import pandas as pd


class evaluateWriter(keras.callbacks.Callback):

    def __init__(
        self,
        items_idx,
        sbert,
        texts,
        evaluator,
        logdir,
        DEVICE,
        sbert_name="sbert_temp_model",
        evaluate_epoch="false",
        save_every_epoch="false",
        eval_every_n_steps: int = 0,
        save_every_n_steps: int = 0,
        eval_model=None,
        coldstart_evaluator=None,
        all_items_idx=None,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.coldstart_evaluator = coldstart_evaluator
        self.logdir = logdir
        self.sbert = sbert
        self.texts = texts
        self.items_idx = items_idx
        self.DEVICE = DEVICE
        self.results_list = []
        self.sbert_name = sbert_name
        self.evaluate_epoch = evaluate_epoch
        self.save_every_epoch = save_every_epoch
        self.eval_every_n_steps = int(eval_every_n_steps)
        self.save_every_n_steps = int(save_every_n_steps)
        self.eval_model = eval_model
        self.global_step = 0
        self.all_items_idx = all_items_idx

    def _evaluate_and_save(self, *, suffix: str, eval_model=None, **kwargs):
        eval_model = eval_model if eval_model is not None else self.eval_model
        do_save = bool(kwargs.pop("_do_save", False))
        do_eval = bool(kwargs.pop("_do_eval", False))
        if eval_model is None:
            raise ValueError("evaluateWriter requires eval_model for save/eval operations.")

        if do_save:
            save_fn = getattr(eval_model, "save", None)
            if callable(save_fn):
                model_path = f"{self.sbert_name}-{suffix}"
                save_fn(model_path)
            else:
                raise ValueError("evaluateWriter: eval_model has no callable save(path) method.")

        if not do_eval:
            return
        df_preds = eval_model.predict_df(
            self.evaluator.test_src,
            texts=self.texts,
            items_idx=self.items_idx,
            **kwargs,
        )
        results = self.evaluator(df_preds)

        if hasattr(eval_model, "predict_df_encoder"):
            df_preds_encoder = eval_model.predict_df_encoder(
                self.evaluator.test_src,
                texts=self.texts,
                items_idx=self.all_items_idx,
                **kwargs,
            )
            results_encoder = self.evaluator(df_preds_encoder)
            results_encoder = {("encoder_" + k): v for k, v in results_encoder.items()}
            results.update(results_encoder)

            if self.coldstart_evaluator is not None:
                df_preds_coldstart = eval_model.predict_df_encoder(
                    self.coldstart_evaluator.test_src,
                    texts=self.texts,
                    items_idx=self.all_items_idx,
                    **kwargs,
                )
                coldstart_results = self.coldstart_evaluator(df_preds_coldstart)
                coldstart_results = {("encoder_cold_start_" + k): v for k, v in coldstart_results.items()}
                results.update(coldstart_results)

        print(results)
        pd.Series(results).to_csv(f"{self.logdir}/result-{suffix}.csv")
        self.results_list.append(results)

    def on_epoch_end(self, epoch, logs=None, eval_model=None, **kwargs):
        do_save = self.save_every_epoch == "true"
        do_eval = self.evaluate_epoch == "true"
        if do_save or do_eval:
            self._evaluate_and_save(
                suffix=f"epoch-{epoch}",
                eval_model=eval_model,
                _do_save=do_save,
                _do_eval=do_eval,
                **kwargs,
            )

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1
        do_save = (self.save_every_n_steps > 0) and (self.global_step % self.save_every_n_steps == 0)
        do_eval = (self.eval_every_n_steps > 0) and (self.global_step % self.eval_every_n_steps == 0)
        self._evaluate_and_save(
            suffix=f"step-{self.global_step}",
            _do_save=do_save,
            _do_eval=do_eval,
        )


class L3AEFinalizeBOnTrainEnd(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        m = self.model
        m.l3ae_refresh_b()
