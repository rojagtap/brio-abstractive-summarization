from functools import partial

import torch
from nltk import word_tokenize, sent_tokenize
from rouge_score.rouge_scorer import RougeScorer
from tqdm import tqdm

from config import CNNDMConfig, XSumConfig
from data import BrioDataset, pad_sequences
from loss import LabelSmoothingLoss, RankingLoss
from model import BRIO
from utils import IO


def train(gpu, train_path, val_path, save_path, dataset):
    if dataset == "cnndm":
        params = CNNDMConfig()
    elif dataset == "xsum":
        params = XSumConfig()
    else:
        raise NotImplementedError("Unknown dataset %s" % dataset)

    train_set = BrioDataset(params.model_name, train_path, encoder_maxlen=params.encoder_maxlen, decoder_maxlen=params.decoder_maxlen, n_candidates=params.n_candidates)
    val_set = BrioDataset(params.model_name, val_path, encoder_maxlen=params.encoder_maxlen, decoder_maxlen=params.decoder_maxlen, n_candidates=params.n_candidates, testing=True)

    train_padding_fn = partial(pad_sequences, pad_token_id=train_set.tokenizer.pad_token_id, testing=False)
    val_padding_fn = partial(pad_sequences, pad_token_id=val_set.tokenizer.pad_token_id, testing=True)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=params.train_batch_size, shuffle=True, num_workers=4, collate_fn=train_padding_fn)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=params.val_batch_size, shuffle=False, num_workers=4, collate_fn=val_padding_fn)
    val_gen_dataloader = torch.utils.data.DataLoader(val_set, batch_size=params.val_gen_batch_size, shuffle=False, num_workers=4, collate_fn=val_padding_fn)

    model = BRIO(train_set.tokenizer.pad_token_id, model_name=params.model_name)
    ranking_loss_fn = RankingLoss()
    mle_loss_fn = LabelSmoothingLoss(ignore_index=train_set.tokenizer.pad_token_id, epsilon=0.1)
    optimizer = torch.optim.Adam(model.parameters())

    io = IO(base_dir=save_path, checkpoint_dir='checkpoints', logdir='logs')

    model = model.to(gpu)

    model.train()
    model.scoring_mode()

    min_ranking_loss = min_mle_loss = 1e5
    steps, epoch = io.load(model, optimizer)
    for epoch in range(epoch, params.n_epochs):
        avg_loss = avg_mle_loss = avg_ranking_loss = 0

        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            encoder_inputs_batch = batch["encoder_inputs"].to(gpu)
            decoder_inputs_batch = batch["decoder_inputs"].to(gpu)

            output = model(encoder_inputs_batch, decoder_inputs_batch, length_penalty=params.length_penalty)
            candidate_scores, summary_score, logits = output["candidate_scores"], output["summary_score"], output["probs"][:, :-1]

            ranking_loss = ranking_loss_fn(candidate_scores, summary_score)
            mle_loss = mle_loss_fn(logits.transpose(1, 2), decoder_inputs_batch[:, 0, 1:])

            loss = 10 * ranking_loss + 0.1 * mle_loss

            avg_loss += loss.item() / params.accumulation_steps
            avg_mle_loss += mle_loss.item() / params.accumulation_steps
            avg_ranking_loss += ranking_loss.item() / params.accumulation_steps

            loss = loss / params.accumulation_steps
            loss.backward()

            # accumulate losses in model.trainable_param.grad at every step
            # and apply after 'accumulation_steps' steps
            if (step + 1) % params.accumulation_steps == 0:
                steps += 1

                # adjust learning rate
                lr = 2e-3 * min(steps ** (-0.5), steps * (10000 ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()

            # report stats every 100 steps
            if (step + 1) % 100 == 0:
                io.plot("loss", {"loss": avg_loss / 100}, steps)
                io.plot("mle_loss", {"loss": avg_mle_loss / 100}, steps)
                io.plot("ranking_loss", {"loss": avg_ranking_loss / 100}, steps)

                avg_loss = avg_mle_loss = avg_ranking_loss = 0

            del candidate_scores, summary_score, logits, loss, mle_loss, ranking_loss, output

            # validation every 1000 steps
            if (steps + 1) % 1000 == 0:
                result = validate(gpu, val_dataloader, val_gen_dataloader, model, val_set.tokenizer, params)

                mle_loss = params.eval(result["gen_rouge1"], result["gen_rouge2"], result["gen_rougelsum"])
                ranking_loss = params.eval(result["score_rouge1"], result["score_rouge2"], result["score_rougelsum"])

                if mle_loss < min_mle_loss:
                    min_mle_loss = mle_loss
                    io.save(model, "generation_best.bin")

                if ranking_loss < min_ranking_loss:
                    min_ranking_loss = ranking_loss
                    io.save(model, "scoring_best.bin")


def validate(gpu, val_dataloader, val_gen_dataloader, model, tokenizer, params):
    score_rouge1 = score_rouge2 = score_rougelsum = 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    mle_loss_fn = LabelSmoothingLoss(ignore_index=tokenizer.pad_token_id, epsilon=0.1)

    model.eval()
    model.scoring_mode()

    n_samples = 0
    with torch.no_grad(), tqdm(val_dataloader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            tepoch.set_description(f"val score step {step}")

            samples = batch["data"]
            encoder_inputs_batch = batch["encoder_inputs"].to(gpu)
            decoder_inputs_batch = batch["decoder_inputs"].to(gpu)

            output = model(encoder_inputs_batch, decoder_inputs_batch, length_penalty=params.length_penalty)
            candidate_scores, summary_score, logits = output["candidate_scores"].cpu().numpy(), output["summary_score"], output["probs"][:, :-1]

            mle_loss = mle_loss_fn(logits.transpose(1, 2), decoder_inputs_batch[:, 0, 1:])

            # get the highest prediction score (best) candidate
            max_ids = candidate_scores.argmax(1)
            for i in range(candidate_scores.shape[0]):
                n_samples += 1

                best_candidate = samples[i]["candidates"][max_ids[i]][0]
                score = rouge_scorer.score("\n".join(samples[i]["abstract"]), "\n".join(best_candidate))

                score_rouge1 += score["rouge1"].fmeasure
                score_rouge2 += score["rouge2"].fmeasure
                score_rougelsum += score["rougeLsum"].fmeasure

            tepoch.set_postfix(rouge1=score_rouge1 / n_samples, rouge2=score_rouge2 / n_samples, rougelsum=score_rougelsum / n_samples)

    score_rouge1 = score_rouge1 / n_samples
    score_rouge2 = score_rouge2 / n_samples
    score_rougelsum = score_rougelsum / n_samples
    mle_loss = mle_loss / n_samples

    gen_rouge1 = gen_rouge2 = gen_rougelsum = 0

    model.generation_mode()

    n_samples = 0
    with torch.no_grad(), tqdm(val_gen_dataloader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            tepoch.set_description(f"val gen step {step}")

            articles = [" ".join(sample["article_untok"]) for sample in batch["data"]]
            inputs = tokenizer.batch_encode_plus(articles, max_length=params.encoder_maxlen, return_tensors="pt", padding="max_length", truncation=True)

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(gpu),
                attention_mask=inputs["attention_mask"].to(gpu),
                max_length=params.decoder_maxlen + 2,   # +2 from original because we start at step=1 and stop before max_length
                min_length=params.decoder_minlen + 1,   # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                num_beams=params.n_beams,
                length_penalty=params.length_penalty,
                early_stopping=True,
            )

            summaries = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False) for pred in summaries]
            for (pred, real) in zip(summaries, batch["data"]):
                n_samples += 1

                pred = pred.replace("\n", " ")
                abstract = " ".join(real["abstract_untok"])

                score = rouge_scorer.score(
                    "\n".join(sent_tokenize(" ".join(word_tokenize(abstract.strip())))),
                    "\n".join(sent_tokenize(" ".join(word_tokenize(pred.strip()))))
                )

                gen_rouge1 += score["rouge1"].fmeasure
                gen_rouge2 += score["rouge2"].fmeasure
                gen_rougelsum += score["rougeLsum"].fmeasure

            tepoch.set_postfix(rouge1=gen_rouge1 / n_samples, rouge2=gen_rouge2 / n_samples, rougelsum=gen_rougelsum / n_samples)

    gen_rouge1 = gen_rouge1 / n_samples
    gen_rouge2 = gen_rouge2 / n_samples
    gen_rougelsum = gen_rougelsum / n_samples

    return {
        "mle_loss": mle_loss,
        "score_rouge1": score_rouge1,
        "score_rouge2": score_rouge2,
        "score_rougelsum": score_rougelsum,
        "gen_rouge1": gen_rouge1,
        "gen_rouge2": gen_rouge2,
        "gen_rougelsum": gen_rougelsum
    }
