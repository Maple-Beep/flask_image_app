# ========================================================================
# flask_image_app/inference_engine/vocabulary.py
# 词汇表管理
# ========================================================================

import pickle

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def idx_to_word(self, idx):
        return self.idx2word.get(idx, '<UNK>')