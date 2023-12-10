class Config:
    model_name = "facebook/bart-large"
    n_candidates = 16
    n_epochs = 100
    val_batch_size = 1
    val_gen_batch_size = 8


class CNNDMConfig(Config):
    encoder_maxlen = 1024
    decoder_maxlen = 128
    generation_maxlen = 140
    generation_minlen = 55
    train_batch_size = 1
    accumulation_steps = 8
    length_penalty = 2.0
    n_beams = 4

    def eval(self, rouge1, rouge2, rougelsum):
        return 1 - (rouge1 * rouge2 + rougelsum) / 3


class XSumConfig(Config):
    encoder_maxlen = 512
    decoder_maxlen = 80
    generation_maxlen = 62
    generation_minlen = 11
    train_batch_size = 2
    accumulation_steps = 4
    length_penalty = 0.6
    n_beams = 8

    def eval(self, rouge1, rouge2, rougelsum):
        return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
