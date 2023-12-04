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
    def __init__(self, margin=0.001, gold_margin=0, gold_weight=0):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.gold_margin = gold_margin
        self.gold_weight = gold_weight

    def forward(self, scores, summary_score):
        ones = torch.ones_like(scores)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        totalloss = loss_func(scores, scores, ones)
        # candidate loss
        for i in range(1, scores.size(1)):
            positive_score = scores[:, :-i]
            negative_score = scores[:, i:]
            positive_score = positive_score.contiguous().view(-1)
            negative_score = negative_score.contiguous().view(-1)
            ones = torch.ones_like(positive_score)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            loss = loss_func(positive_score, negative_score, ones)
            totalloss += loss

        # predicted summary loss
        positive_score = summary_score.unsqueeze(-1).expand_as(scores)
        negative_score = scores
        positive_score = positive_score.contiguous().view(-1)
        negative_score = negative_score.contiguous().view(-1)
        ones = torch.ones_like(positive_score)
        loss_func = torch.nn.MarginRankingLoss(self.gold_margin)
        totalloss += self.gold_weight * loss_func(positive_score, negative_score, ones)
        return totalloss
