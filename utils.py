import torch
import torch.functional as F



def quantile_loss(pred, target, quantiles):
    """
    Calculate quantile loss over all dimensions.
    :param pred: prediction for quantiles                     # dimensions: (batch, len(hidden_states), horizon, len(quantiles))
    :param target: real values, expanded to match pred shape  # dimensions: (batch, len(hidden_states), horizon, len(quantiles))
    :param quantiles: list of quantiles                       # dimensions: len(quantiles)
    :return: total loss on batch_size, T, K, Q
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    target = target.unsqueeze(1)  # dimensions: (batch, 1, horizon, 1)
    target = target.expand(*pred.shape).to(device)  # dimensions: (batch, len(hidden_states), horizon, len(quantiles))

    quantiles = torch.Tensor(quantiles).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)  # dimensions: (1, 1, 1, len(quantiles))
    quantiles = quantiles.expand(pred.shape).to(device)
    pred=pred.to(device)

    relu = torch.nn.ReLU()
    loss = quantiles * relu(target - pred) + (1 - quantiles) * relu(pred - target)
    loss = torch.mean(loss)
    return loss
