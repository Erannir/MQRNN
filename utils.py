import torch

def quantile_loss(pred, target, quantiles):
    """
    Calculate quantile loss over all dimensions.
    :param pred: prediction for quantiles                     # dimensions: (batch, seq_len, horizon, num_quantiles)
    :param target: real values, expanded to match pred shape  # dimensions: (batch, seq_len, horizon, 1)
    :param quantiles: list of quantiles                       # dimensions: num_quantiles
    :return: total loss on batch_size, T, K, Q
    """
    #target = target.unsqueeze(1)  # dimensions: (batch, 1, horizon, 1)

    target = target.expand(*pred.shape)  # dimensions: (batch, len(hidden_states), horizon, len(quantiles))

    quantiles = torch.Tensor(quantiles).view(*((len(pred.shape)-1) * [1]), -1)  # unsqueezing 2/3 times
    quantiles = quantiles.expand(pred.shape).to(device)

    relu = torch.nn.ReLU()
    loss = quantiles * relu(target - pred) + (1 - quantiles) * relu(pred - target)
    loss = torch.mean(loss)
    return loss
