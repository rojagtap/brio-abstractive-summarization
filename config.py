class Config:
    n_epochs = 100                                      # number of epochs
    n_warmup_steps = 10000                              # number of warmup steps
    n_candidates = 16                                   # number of candidates in the sequence
    val_batch_size = 1                                  # validation set batch size
    val_gen_batch_size = 8                              # validation set for generation batch size
    summary_score_ranking_weight = 0                    # weight of gold sample ranking loss (original summary) in total ranking loss, i.e., total_ranking_loss = ranking_loss_for_candidates + summary_score_ranking_weight * ranking_loss_for_gold_sample
    summary_score_ranking_loss_margin = 0               # ranking loss of gold sample will be 0 when the distance between the gold sample and candidates will be > summary_score_ranking_loss_margin. note that the distance here is calculated based on scores, so margin = 0 means, the loss will be 0 when the score of the actual summary is ranked higher than the candidate summaries
    candidate_scores_ranking_loss_margin = 0.001        # ranking loss of any given candidate sample will be 0 when the distance between this sample and a lower ranked sample will be > candidate_scores_ranking_loss_margin. note that the distance here is calculated based on scores, so margin = 0.001 means, the loss will be 0 when the score of the given candidate is scored higher than lower ranked candidate summaries by a 'margin' of 0.001
    mle_loss_weight = 0.1                               # weight of mle loss in total loss, i.e., total_loss = mle_loss_weight * mle_loss + ranking_loss_weight * ranking_loss
    ranking_loss_weight = 10                            # weight of ranking loss in total loss, i.e., total_loss = mle_loss_weight * mle_loss + ranking_loss_weight * ranking_loss
    label_smoothing_epsilon = 0.1                       # for a label vector [0, 0, 0, 1], use [0.025, 0.025, 0.025, 0.925] (with epsilon = 0.1)
    grad_norm_clipping = 0                              # the L2 norm of the gradient vector cannot exceed this value. if it does, then the gradient is scaled to this value (used with gradient clipping)
    max_learning_rate = 2e-3                            # learning rate will decay from this value
    use_log_softmax = True                              # whether to use logarithmic-scale softmax or vanilla softmax


class CNNDMConfig(Config):
    model_name = "facebook/bart-large-cnn"              # which model to use
    n_beams = 4                                         # number of beans for generation
    train_batch_size = 1                                # training set batch size
    length_penalty = 2.0                                # is used for length normalization of log probabilities
    loss_accumulation_steps = 8                         # loss is accumulated for these many steps before finally weights are updated
    encoder_maxlen = 1024                               # maximum allowed length for the paragraph to be summarized
    decoder_maxlen = 128                                # maximum allowed length for summary
    generation_maxlen = 140                             # maximum allowed length for summary
    generation_minlen = 55                              # generation step to generate a summary this long

    @staticmethod
    def eval(rouge1, rouge2, rougelsum):
        return 1 - (rouge1 * rouge2 + rougelsum) / 3


class XSumConfig(Config):
    model_name = "google/pegasus-xsum"                  # which model to use
    n_beams = 8                                         # number of beans for generation
    train_batch_size = 2                                # training set batch size
    length_penalty = 0.6                                # is used for length normalization of log probabilities
    loss_accumulation_steps = 4                         # loss is accumulated for these many steps before finally weights are updated
    encoder_maxlen = 512                                # maximum allowed length for the paragraph to be summarized
    decoder_maxlen = 80                                 # maximum allowed length for summary
    generation_maxlen = 62                              # maximum allowed length for summary
    generation_minlen = 11                              # generation step to generate a summary this long

    @staticmethod
    def eval(rouge1, rouge2, rougelsum):
        return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
