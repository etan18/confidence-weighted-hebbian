import logging
import torch
from .learning_rule import LearningRule

class RobustHebbianLearningRule(LearningRule):

    def __init__(self, c=0.001):
        """
        Args:
            c: The learning rate (eta) for the rule.
        """
        super().__init__()
        self.c = c

        # Running statistics for confidence computation
        self.initialized_stats = False
        self.running_mean = None
        self.running_var = None
        self.count = 0

    def _initialize_stats(self, out_dim):
        # Initialize statistics for the output dimension
        self.running_mean = torch.zeros(out_dim)
        self.running_var = torch.ones(out_dim)   # Start var at 1 to avoid div by zero
        self.count = 0
        self.initialized_stats = True

    def _update_stats(self, Y):
        # Y is (batch_size, out_dim)
        batch_mean = torch.mean(Y, dim=0)
        batch_var = torch.var(Y, dim=0, unbiased=False)  # population variance

        self.count += 1
        # Update running mean and var
        self.running_mean = self.running_mean + (batch_mean - self.running_mean) / self.count
        self.running_var = self.running_var + (batch_var - self.running_var) / self.count

    def update(self, inputs, w):
        """
        Perform the robust Hebbian update (modified Sanger's rule).
        
        Args:
            inputs (torch.Tensor): Shape (batch_size, in_dim)
            w (torch.Tensor): Shape (out_dim, in_dim)
        
        Returns:
            torch.Tensor: The weight update Δw of shape (out_dim, in_dim)
        """
        batch_size, in_dim = inputs.shape
        out_dim = w.shape[0]

        if not self.initialized_stats:
            self._initialize_stats(out_dim)

        # Compute all y-values for the current batch to update statistics
        # y = w * x (in a linear form: y = w x^T but here x is row vector)
        # But we must do per-sample according to style in OjasRule
        # We'll store them in a tensor for stats
        Ys = torch.zeros(batch_size, out_dim)

        # First pass: compute all Ys and store them
        for idx in range(batch_size):
            x = inputs[idx]
            # y = w * x^T
            y = torch.mm(w, x.unsqueeze(1))  # (out_dim, 1)
            y = y.squeeze(1)  # (out_dim,)
            Ys[idx] = y

        # Update statistics with the entire batch
        self._update_stats(Ys)

        # Compute confidence c_i = 1/(1+Var(y_i))
        c_i = 1.0 / (1.0 + self.running_var)

        # Now compute Δw according to modified Sanger's rule
        # Δw_ij = η * c_i * Σ_over_batch[y_i (x_j - Σ_{k<i} y_k w_kj)]
        # We'll follow a structure similar to Oja, iterating over samples and accumulating
        d_ws = torch.zeros(batch_size, *w.shape)

        for idx in range(batch_size):
            x = inputs[idx]
            y = Ys[idx]

            # For each neuron i
            d_w = torch.zeros_like(w)
            for i in range(out_dim):
                # Compute the projection: sum_{k<i} y_k w_k
                if i == 0:
                    proj = torch.zeros(in_dim)
                else:
                    # y[:i] is (i,), w[:i, :] is (i, in_dim)
                    proj = torch.mv(w[:i, :].t(), y[:i])

                residual = x - proj
                # Δw_i = η * c_i * y_i * residual
                d_w[i, :] = self.c * c_i[i] * y[i] * residual

            d_ws[idx] = d_w

        # Return the mean update over the batch
        return torch.mean(d_ws, dim=0)
