import numpy as np
import torch
from scipy.stats import norm


def predict_distribution(model, x_input, y_range):
    """Return probability density over y_range for a single input"""
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(torch.tensor(x_input, dtype=torch.float32).unsqueeze(0))
    pi, mu, sigma = pi[0].numpy(), mu[0].numpy(), sigma[0].numpy()

    pdf = np.zeros_like(y_range)
    for k in range(len(pi)):
        pdf += pi[k] * norm.pdf(y_range, mu[k], sigma[k])
    return pdf, (pi, mu, sigma)
