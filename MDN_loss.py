import torch
import torch.optim as optim
import numpy as np


def mdn_loss_crps(pi, mu, sigma, y):
    """
    Compute CRPS (Continuous Ranked Probability Score) for a Gaussian mixture, computed
    analytically as:

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]

    where X and X' are independent draws from the predicted distribution.
    """
    eps = 1e-12
    batch_size = y.shape[0]
    n_components = pi.shape[1]
    
    # Ensure sigma is positive and well-behaved
    sigma = torch.clamp(sigma, min=1e-6, max=1e4)
    
    # Expand y to match component dimensions [batch_size, n_components]
    y_expanded = y.unsqueeze(1).expand_as(mu)
    
    # First term: E[|X - y|] for Gaussian mixture
    # For a single Gaussian: E[|X - y|] = 2*sigma*phi((y-mu)/sigma) + (y-mu)*[2*Phi((y-mu)/sigma) - 1]
    # where phi is the standard normal PDF and Phi is the standard normal CDF
    
    z = (y_expanded - mu) / (sigma + eps)  # standardized values
    
    # Standard normal PDF: phi(z) = (1/sqrt(2*pi)) * exp(-0.5*z^2)
    phi_z = (1.0 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * z**2)
    
    # Standard normal CDF: Phi(z) using error function
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    sqrt_2 = np.sqrt(2.0)
    Phi_z = 0.5 * (1.0 + torch.erf(z / sqrt_2))
    
    # First term per component
    first_term_components = 2.0 * sigma * phi_z + (y_expanded - mu) * (2.0 * Phi_z - 1.0)
    
    # Weight by mixture probabilities
    first_term = torch.sum(pi * first_term_components, dim=1)  # [batch_size]
    
    # Second term: 0.5 * E[|X - X'|] for mixture
    # This involves pairwise differences between components
    second_term_sum = 0.0
    
    for i in range(n_components):
        for j in range(n_components):
            # Get parameters for components i and j
            pi_i = pi[:, i]  # [batch_size]
            pi_j = pi[:, j]  # [batch_size]
            mu_i = mu[:, i]
            mu_j = mu[:, j]
            sigma_i = sigma[:, i]
            sigma_j = sigma[:, j]
            
            # For two Gaussians N(mu_i, sigma_i^2) and N(mu_j, sigma_j^2):
            # E[|X_i - X_j|] = sqrt(2/pi) * sqrt(sigma_i^2 + sigma_j^2) * exp(-0.5 * (mu_i - mu_j)^2 / (sigma_i^2 + sigma_j^2))
            #                  + (mu_i - mu_j) * [2*Phi((mu_i - mu_j)/sqrt(sigma_i^2 + sigma_j^2)) - 1]
            
            sigma_sum_sq = sigma_i**2 + sigma_j**2 + eps
            sigma_sum = torch.sqrt(sigma_sum_sq)
            mu_diff = mu_i - mu_j
            
            z_ij = mu_diff / (sigma_sum + eps)
            
            # exp term
            exp_term = torch.exp(-0.5 * z_ij**2)
            
            # CDF term
            Phi_z_ij = 0.5 * (1.0 + torch.erf(z_ij / sqrt_2))
            
            # E[|X_i - X_j|]
            exp_abs_diff = np.sqrt(2.0 / np.pi) * sigma_sum * exp_term + mu_diff * (2.0 * Phi_z_ij - 1.0)
            
            # Weight by pi_i * pi_j
            second_term_sum = second_term_sum + pi_i * pi_j * exp_abs_diff
    
    second_term = 0.5 * second_term_sum  # [batch_size]
    
    # CRPS = first_term - second_term
    crps = first_term - second_term
    
    # Return mean CRPS over batch
    return crps.mean()


# =========================
# Loss Function Wrapper
# =========================
def mdn_loss(pi, mu, sigma, y, loss_type='crps'):
    """
    Wrapper for MDN loss function.
    
    Args:
        pi: mixture weights [batch_size, n_components]
        mu: component means [batch_size, n_components]
        sigma: component standard deviations [batch_size, n_components]
        y: target values [batch_size]
        loss_type: 'crps' or 'nll'
    
    Returns:
        loss: scalar loss value
    """
    if loss_type == 'crps':
        return mdn_loss_crps(pi, mu, sigma, y)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'crps' or 'nll'.")
