import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import sentence_transformers

from keras.layers import TorchModuleWrapper


class LayerSBERT(keras.layers.Layer):

    def __init__(self, model, device, tokenized_sentences):
        super().__init__()
        self.device = device
        self.sbert = TorchModuleWrapper(model.to(device))
        self.tokenize_ = self.sb().tokenize
        self.tokenized_sentences = tokenized_sentences
        self.build()

    def sb(self):
        for module in self.sbert.modules():
            if isinstance(module, sentence_transformers.SentenceTransformer):
                return module
        raise RuntimeError("SentenceTransformer module not found in LayerSBERT.")

    def parameters(self, recurse=True):
        return self.sbert.parameters()

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable(initializer=param, trainable=param.requires_grad)
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def tokenize(self, inp):
        return {k: v.to(self.device) for k, v in self.tokenize_(inp).items()}

    def build(self):
        self.to(self.device)
        sample_input = {k: v[:2].to(self.device) for k, v in self.tokenized_sentences.items()}
        _ = self.call(sample_input)
        self.track_module_parameters()

    def call(self, x, training=None):
        return self.sbert(x, training=training)["sentence_embedding"]
