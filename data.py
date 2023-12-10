import json
import os

import torch
from transformers import BartTokenizer


def pad_sequences(batch, pad_token_id, testing=False):
    def pad(sequences, maxlen):
        """
        pad each sequence to max length across batch
        """
        padded = torch.ones(len(sequences), maxlen, dtype=sequences[0].dtype) * pad_token_id
        for (i, sequence) in enumerate(sequences):
            padded[i, :sequence.size(0)] = sequence

        return padded

    encoder_batch, decoder_batch = [], []
    for sample in batch:
        encoder_batch.append(sample["encoder_inputs"])
        decoder_batch.append(sample["decoder_inputs"])

    encoder_maxlen = max(sample.size(0) for sample in encoder_batch)
    decoder_maxlen = max(max(len(candidate) for candidate in sample) for sample in decoder_batch)

    inputs = {
        "encoder_inputs": pad(encoder_batch, encoder_maxlen),
        "decoder_inputs": torch.stack([pad(sample, decoder_maxlen) for sample in decoder_batch]),
    }

    if testing:
        inputs["data"] = [sample["data"] for sample in batch]

    return inputs


class BrioDataset(torch.utils.data.Dataset):
    def __init__(self, model_name, dataset_dir, encoder_maxlen, decoder_maxlen, n_candidates, testing=False, is_sorted=True, is_untok=True):
        self.testing = testing
        self.is_untok = is_untok
        self.is_sorted = is_sorted
        self.dataset_dir = dataset_dir
        self.n_candidates = n_candidates
        self.encoder_maxlen = encoder_maxlen
        self.decoder_maxlen = decoder_maxlen
        self.size = len(os.listdir(self.dataset_dir))
        self.tokenizer = BartTokenizer.from_pretrained(model_name, verbose=False)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_dir, "%d.json" % idx), "r") as fp:
            data = json.load(fp)

        article = data["article_untok"] if self.is_untok else data["article"]
        abstract = data["abstract_untok"] if self.is_untok else data["abstract"]
        candidates = data["candidates_untok"] if self.is_untok else data["candidates"]

        if self.is_sorted:
            # sort based on scores
            candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)

        candidates = candidates[:self.n_candidates]

        encoder_inputs = self.tokenizer.batch_encode_plus([" ".join(article)], max_length=self.encoder_maxlen, return_tensors="pt", padding=False, truncation=True)["input_ids"].squeeze(0)
        decoder_inputs = self.tokenizer.batch_encode_plus([" ".join(abstract)] + [" ".join(x[0]) for x in candidates], max_length=self.decoder_maxlen, return_tensors="pt", padding=True, truncation=True)["input_ids"]

        inputs = {
            "encoder_inputs": encoder_inputs,
            "decoder_inputs": decoder_inputs,
        }

        if self.testing:
            inputs["data"] = data

        return inputs
