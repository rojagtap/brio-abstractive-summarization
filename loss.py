import torch


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon

    def forward(self, inputs, target):
        inputs = inputs.transpose(1, 2)  # [batch_size, seq_len, word_num]
        inputs = torch.log_softmax(inputs, dim=2)

        k = inputs.size(2)
        target_prob = torch.ones_like(inputs).type_as(inputs) * self.epsilon * 1 / k

        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))

        loss = - torch.mul(target_prob, inputs)
        loss = loss.sum(2)

        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(inputs)
        loss = (torch.mul(loss, mask).sum() / mask.sum()).mean()

        return loss


class RankingLoss(torch.nn.Module):
    def __init__(self, candidate_ranking_margin=0.001, summary_ranking_margin=0, summary_ranking_loss_weight=0):
        super(RankingLoss, self).__init__()
        self.candidate_ranking_margin = candidate_ranking_margin
        self.summary_ranking_margin = summary_ranking_margin
        self.summary_ranking_loss_weight = summary_ranking_loss_weight

    def forward(self, candidate_scores, summary_score):
        ones = torch.ones_like(candidate_scores)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(candidate_scores, candidate_scores, ones)

        # candidate loss
        for i in range(1, candidate_scores.size(1)):
            high_rank = candidate_scores[:, :-i]
            low_rank = candidate_scores[:, i:]
            high_rank = high_rank.contiguous().view(-1)
            low_rank = low_rank.contiguous().view(-1)
            ones = torch.ones_like(high_rank)
            loss_func = torch.nn.MarginRankingLoss(self.candidate_ranking_margin * i)       # λij = λ * rank difference
            loss = loss_func(high_rank, low_rank, ones)
            total_loss += loss

        # predicted summary loss
        high_rank = summary_score.unsqueeze(-1).expand_as(candidate_scores)
        low_rank = candidate_scores
        high_rank = high_rank.contiguous().view(-1)
        low_rank = low_rank.contiguous().view(-1)
        ones = torch.ones_like(high_rank)
        loss_func = torch.nn.MarginRankingLoss(self.summary_ranking_margin)
        total_loss += self.summary_ranking_loss_weight * loss_func(high_rank, low_rank, ones)

        return total_loss
