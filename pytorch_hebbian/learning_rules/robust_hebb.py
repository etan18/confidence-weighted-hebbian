import logging
import math
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from .learning_rule import LearningRule

class RobustHebbsRule(LearningRule):
    def __init__(self, c=0.0001, momentum=0.9):
        super().__init__()
        self.c = c
        self.momentum = momentum
        self.prev_update = None

        # Running statistics for (x, y)
        self.mean_x = None
        self.var_x = 1.0  # Shared variance for simplicity
        self.count = 0
        self.norm = 2

    def _update_statistics(self, inputs, labels):
        """
        Update the running mean and variance of flattened inputs and labels.

        Args:
            inputs (torch.Tensor): Input features of shape (batch_size, in_dim).
            labels (torch.Tensor): Labels of shape (batch_size,) or (batch_size, out_dim).
        """
        batch_data = torch.cat([inputs, labels], dim=1)  # Concatenate inputs and labels
        batch_mean = batch_data.mean(dim=0)  # Batch mean
        batch_var = batch_data.var(dim=0, unbiased=False).mean()  # Batch variance (shared scalar)

        # Update mean and variance
        if self.mean_x is None:
            self.mean_x = batch_mean
            self.var_x = batch_var
            self.count = batch_data.size(0)
        else:
            prev_count = self.count
            total_count = prev_count + batch_data.size(0)
            
            # Update mean
            delta = batch_mean - self.mean_x
            self.mean_x += (batch_data.size(0) / total_count) * delta
            
            # Update variance
            self.var_x = ((prev_count * self.var_x + batch_data.size(0) * batch_var) / total_count)
            self.count = total_count

    def compute_confidence(self, inputs, labels):
        """
        Compute Bayesian confidence for each point (x, y).

        Args:
            inputs (torch.Tensor): Input features of shape (batch_size, in_dim).
            labels (torch.Tensor): Labels of shape (batch_size,) or (batch_size, out_dim).

        Returns:
            torch.Tensor: Confidence scores of shape (batch_size,).
        """
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        z = torch.cat([inputs, labels], dim=1)  # Concatenate inputs and labels
        delta_z = z - self.mean_x  # Difference from mean

        # Compute Gaussian likelihood
        exponent = -0.5 * (delta_z ** 2).sum(dim=1) / self.var_x
        likelihood = torch.exp(exponent) / ((2 * math.pi * self.var_x) ** (0.5 * z.size(1)))

        # Normalize to obtain posterior confidence
        confidence = likelihood / likelihood.sum()
        return confidence

    def update(self, inputs, labels, w):
        """
        Perform a Bayesian robust Hebbian update.

        Args:
            inputs (torch.Tensor): Shape (batch_size, in_dim).
            labels (torch.Tensor): Shape (batch_size,) or (batch_size, 1).
            w (torch.Tensor): Shape (out_dim, in_dim).

        Returns:
            torch.Tensor: The weight update Δw of shape (out_dim, in_dim).
        """
        batch_size = labels.shape[0]
        in_dims = inputs.numel() // batch_size
        #reshaped_inputs = inputs.view(batch_size, in_dims)  # Flatten inputs if needed
        reshaped_inputs = inputs.view(batch_size, -1)
        reshaped_labels = labels.view(batch_size, -1)  # Ensure labels are 2D

        # Update running statistics
        self._update_statistics(reshaped_inputs, reshaped_labels)

        # Compute Bayesian confidence
        c_i = self.compute_confidence(reshaped_inputs, reshaped_labels)  # Shape: (batch_size,)

        # Compute residuals and weight updates
        with autocast():
            print(f"reshaped inputs unsqueeze: {reshaped_inputs.unsqueeze(1).shape}, reshaped inputs: {reshaped_inputs.shape}, weight.t: {w.t().shape}, weight: {w.shape}")
            print(f"matmul 1: {torch.mm(reshaped_inputs, w.t()).shape}")
            print(f"matmul 2: {torch.mm(torch.mm(reshaped_inputs, w.t()), w).shape} and unsqueezed: {torch.mm(torch.mm(reshaped_inputs, w.t()), w).unsqueeze(1).shape}")
            residuals = reshaped_inputs.unsqueeze(1) - torch.mm(torch.mm(reshaped_inputs, w.t()), w).unsqueeze(1)
            updates = (c_i.view(batch_size, 1, 1) * residuals).mean(dim=0)  # (out_dim, in_dim)
            d_w = updates / (torch.norm(updates, p=2) + 1e-12)  # Normalize update

        # Apply momentum
        if self.prev_update is None:
            self.prev_update = d_w.clone()  # Initialize on the first update
        else:
            self.prev_update.mul_(self.momentum).add_(d_w)  # Update with momentum

        # Use the updated weights for the current step
        return self.c * self.prev_update


class RobustHebbsRule(LearningRule):
    def __init__(self, c=0.0001, momentum=0.9):
        super().__init__()
        self.c = c
        self.momentum = momentum
        self.prev_update = None

        # Running statistics for (x, y)
        self.mean_x = None
        self.var_x = None
        self.mean_y = None
        self.var_y = None
        self.count = 0
        self.norm = 2
        
    def _update_statistics(self, inputs, labels):
        """
        Update the empirical multivariate normal distribution of flattened inputs and labels.

        Args:
            inputs (torch.Tensor): Input features of shape (batch_size, in_dim).
            labels (torch.Tensor): Labels of shape (batch_size,) or (batch_size, out_dim).
        """
        
        # Concatenate along dimension 1
        batch_data = torch.cat([inputs, labels], dim=1)

        # Compute batch mean and covariance
        batch_mean = batch_data.mean(dim=0)  # Shape: (in_dim_flat + out_dim_flat,)
        batch_centered = batch_data - batch_mean  # Shape: (batch_size, in_dim_flat + out_dim_flat)
        batch_cov = (batch_centered.T @ batch_centered) / batch_data.size(0)  # Shape: (in_dim_flat + out_dim_flat, in_dim_flat + out_dim_flat)

        # Initialize or update running statistics
        if self.mean_x is None:
            self.mean_x = batch_mean
            self.cov_x = batch_cov
            self.count = batch_data.size(0)
        else:
            prev_count = self.count
            total_count = prev_count + batch_data.size(0)

            # Update mean
            delta = batch_mean - self.mean_x
            self.mean_x.add_((batch_data.size(0) / total_count) * delta)

            # Update covariance
            self.cov_x = (
                (prev_count * self.cov_x + batch_data.size(0) * batch_cov)
                + (prev_count * batch_data.size(0) / total_count) * (delta.unsqueeze(1) @ delta.unsqueeze(0))
            ) / total_count

            self.count = total_count

    def compute_confidence(self, inputs, labels):
        """
        Compute confidence for each point (x, y) with respect to the stored
        multivariate normal distribution.

        Args:
            inputs (torch.Tensor): Input features of shape (batch_size, in_dim).
            labels (torch.Tensor): Labels of shape (batch_size,) or (batch_size, out_dim).

        Returns:
            torch.Tensor: Confidence scores of shape (batch_size,).
        """
        # Ensure labels is 2D if scalar
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        # Concatenate inputs and labels
        z = torch.cat([inputs, labels], dim=1)  # Shape: (batch_size, in_dim + out_dim)

        # Add small regularization to covariance matrix to prevent singularity
        epsilon = 1e-6
        regularized_cov_x = self.cov_x + epsilon * torch.eye(self.cov_x.shape[0], device=self.cov_x.device)
        
        with autocast():
            try:
                # Use more numerically stable inverse computation
                cov_inv = torch.inverse(regularized_cov_x)
            except RuntimeError:
                # Fallback to identity matrix if inverse fails
                print("Warning: Covariance matrix inversion failed. Using identity matrix.")
                cov_inv = torch.eye(self.cov_x.shape[0], device=self.cov_x.device)

        # Compute delta from mean
        delta_z = z - self.mean_x  # Shape: (batch_size, in_dim + out_dim)

        # Compute Mahalanobis distance
        mahalanobis_sq = torch.einsum('bi,ij,bj->b', delta_z, cov_inv, delta_z)  # Shape: (batch_size,)

        # Compute confidence with a cap to prevent extreme values
        confidence = torch.clamp(torch.exp(-0.5 * mahalanobis_sq), min=1e-10, max=1.0)  # Shape: (batch_size,)
        confidence = F.softmax(confidence, dim=0)
        return confidence
        

    def update(self, inputs, labels, w):
        """
        Perform a robust Hebbian update using confidence-based weighting.
        
        Args:
            inputs (torch.Tensor): Shape (batch_size, in_dim)
            labels (torch.Tensor): Shape (batch_size,) or (batch_size, 1)
            w (torch.Tensor): Shape (out_dim, in_dim)

        Returns:
            torch.Tensor: The weight update Δw of shape (out_dim, in_dim)
        """
        
        batch_size = labels.shape[0]
        in_dims = int(inputs.numel() / batch_size)
        reshaped_inputs = inputs.view(batch_size, in_dims) # 14400 for mnist or mnist fashion
        reshaped_labels = labels.view(batch_size, 1)
        
        # Update running statistics of (x, y) to compute confidence
        self._update_statistics(reshaped_inputs, reshaped_labels)
        # Compute confidence per sample
        c_i = self.compute_confidence(reshaped_inputs, reshaped_labels) # shape: (batch_size,)

        # Compute outputs
        with autocast():
            Y = torch.mm(inputs, w.t())
            residuals = inputs.unsqueeze(1) - torch.mm(torch.mm(inputs, w.t()), w).unsqueeze(1)

            # Compute weight updates using vectorized operations
            updates = c_i.repeat_interleave(576).view(batch_size * 576, 1, 1) * Y.unsqueeze(2) * residuals  # (batch_size, out_dim, in_dim)
            d_w = updates.mean(dim=0)  # (out_dim, in_dim)
            d_w = d_w / (torch.norm(d_w, p=2) + 1e-12) # normalize update

        # Apply momentum
        if self.prev_update is None:
            self.prev_update = d_w.clone()  # Initialize on the first update
        else:
            self.prev_update.mul_(self.momentum).add_(d_w)  # Update with momentum

        # Use the updated weights for the current step
        return self.c * self.prev_update

