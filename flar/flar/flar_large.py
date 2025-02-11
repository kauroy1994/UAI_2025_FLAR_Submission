"""
dagboost_stable.py

The "stable" DAG-Boost method for causal discovery, 
using an augmented Lagrangian for acyclicity 
plus inf/nan avoidance measures (clamping, shrinkage, etc.).

This file is intended to reside inside the 'dagboost/' package folder.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


###############################################################################
# 1) Data simulation & evaluation (with smaller weight_scale by default)
###############################################################################

def generate_erdos_renyi_dag(num_nodes, edge_prob):
    """
    Generate random DAG with given edge probability, ensuring acyclicity
    by linking only forward in a random permutation.
    """
    perm = np.random.permutation(num_nodes)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.rand() < edge_prob:
                adjacency_matrix[perm[i], perm[j]] = 1
    return adjacency_matrix

def generate_scale_free_dag(num_nodes, out_degree=2):
    """
    Scale-Free DAG by preferential attachment. 
    """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    if num_nodes < 2:
        return adjacency_matrix

    adjacency_matrix[0, 1] = 1
    in_degrees = np.zeros(num_nodes)
    in_degrees[1] = 1

    for new_node in range(2, num_nodes):
        prob_attachment = (in_degrees[:new_node] + 1) / np.sum(in_degrees[:new_node] + 1)
        parents = np.random.choice(
            np.arange(new_node),
            size=min(out_degree, new_node),
            replace=False,
            p=prob_attachment
        )
        for p in parents:
            adjacency_matrix[p, new_node] = 1
            in_degrees[new_node] += 1

    return adjacency_matrix

def simulate_sem(adjacency_matrix, n_samples, noise_type='gaussian',
                 weight_scale=0.01,  # small scale to avoid large values
                 random_state=None):
    """
    Simulate linear SEM data: X_j = sum_i (W[i,j]*X_i) + noise_j
    with smaller weight_scale by default (0.01).
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_nodes = adjacency_matrix.shape[0]
    G = nx.DiGraph(adjacency_matrix)
    topo_order = list(nx.topological_sort(G))

    # Random weights
    W = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1:
                W[i, j] = np.random.normal(0, weight_scale)

    X = np.zeros((n_samples, num_nodes))
    for s in range(n_samples):
        for node in topo_order:
            parents = np.where(adjacency_matrix[:, node] == 1)[0]
            parents_sum = np.sum(W[parents, node] * X[s, parents])
            if noise_type == 'gaussian':
                noise = np.random.normal(0,1)
            elif noise_type == 'laplace':
                noise = np.random.laplace(0,1)
            else:
                noise = np.random.exponential(1)
            X[s,node] = parents_sum + noise

    return X

def remove_all_cycles(G):
    """
    Remove edges from cycles until none remain, in-place.
    """
    while True:
        try:
            cycle_edges = nx.find_cycle(G, orientation='original')
            for (u, v, _) in cycle_edges:
                G.remove_edge(u, v)
        except nx.NetworkXNoCycle:
            break
    return G

def shd(true_adj: np.ndarray, est_adj: np.ndarray) -> int:
    """
    Structural Hamming Distance.
    """
    return np.sum(true_adj != est_adj)

def _compute_ancestors(adj: np.ndarray):
    """
    For SID computation, gather ancestors for each node.
    """
    G = nx.DiGraph(adj)
    d = adj.shape[0]
    ancestors_list = []
    for node in range(d):
        an = nx.ancestors(G, node)
        ancestors_list.append(set(an))
    return ancestors_list

def sid(true_adj: np.ndarray, est_adj: np.ndarray) -> int:
    """
    Structural Intervention Distance:
    Count ancestor mismatches across all node pairs.
    """
    true_anc = _compute_ancestors(true_adj)
    est_anc = _compute_ancestors(est_adj)
    d = true_adj.shape[0]
    score = 0
    for j in range(d):
        diff_1 = true_anc[j].difference(est_anc[j])
        diff_2 = est_anc[j].difference(true_anc[j])
        score += len(diff_1) + len(diff_2)
    return score

def compute_total_effect_matrix(W: np.ndarray) -> np.ndarray:
    """
    T = (I - W)^{-1} - I
    """
    d = W.shape[0]
    I = np.eye(d)
    try:
        inv = np.linalg.inv(I - W)
    except np.linalg.LinAlgError:
        return np.zeros((d, d))
    return inv - I

def rmse_ate(W_true: np.ndarray, W_est: np.ndarray) -> float:
    """
    RMSE of total causal effects.
    """
    T_true = compute_total_effect_matrix(W_true)
    T_est  = compute_total_effect_matrix(W_est)
    return np.sqrt(np.mean((T_true - T_est) ** 2))

###############################################################################
# 2) DAG-Boost "stable" version with clamp/shrinkage to avoid inf/nan
###############################################################################

def squared_loss(x_true, x_pred):
    return 0.5 * torch.mean((x_true - x_pred)**2)

def dag_constraint(W):
    """
    h(W) = trace(exp(W*W)) - d
    """
    d = W.shape[0]
    WW = W * W
    expm_WW = torch.matrix_exp(WW)
    h = torch.trace(expm_WW) - d
    return h

def apply_mask(W, mask):
    """
    Zero out disallowed edges (mask==0) and diagonal, in-place.
    """
    with torch.no_grad():
        W *= mask
        d = W.shape[0]
        for i in range(d):
            W[i, i] = 0.0

def binarize_adjacency(W, threshold=0.1):
    """
    Convert real-valued adjacency to binary with the given threshold.
    """
    W_np = W.detach().cpu().numpy()
    W_bin = (np.abs(W_np) > threshold).astype(float)
    np.fill_diagonal(W_bin, 0.0)
    return W_bin

class WeakLearnerNN(nn.Module):
    """
    Small neural net used as a weak learner for a single variable.
    """
    def __init__(self, input_dim, hidden_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class FunctionalBoostingModel(nn.Module):
    """
    Summation of weak learners for each variable i. 
    With optional clamp or partial shrink after training each new learner.
    """
    def __init__(self, d, max_num_weak_learners=2, hidden_dim=4):
        super().__init__()
        self.d = d
        self.max_num_weak_learners = max_num_weak_learners
        self.hidden_dim = hidden_dim
        self.learners_for_var = [[] for _ in range(d)]
        self.weak_learners = nn.ModuleList()
        self.current_counts = [0]*d

    def forward(self, X, W):
        N, d = X.shape
        device = X.device
        Xhat = []
        for i in range(d):
            mask_row = W[i, :]
            masked_input = X * mask_row
            pred_i = torch.zeros((N,1), dtype=X.dtype, device=device)
            for learner in self.learners_for_var[i]:
                # clamp each partial output
                out_i = learner(masked_input)
                out_i = torch.clamp(out_i, -1e4, 1e4)
                pred_i += out_i
            Xhat.append(pred_i)
        return torch.cat(Xhat, dim=1)

    def add_weak_learner(self, i):
        wl = WeakLearnerNN(input_dim=self.d, hidden_dim=self.hidden_dim)
        self.weak_learners.append(wl)
        self.learners_for_var[i].append(wl)
        self.current_counts[i] += 1

    def fit_new_weak_learner(
        self, i, X, residual_i, W,
        n_epochs=5, lr=1e-4, shrinkage=0.1, verbose=False
    ):
        """
        1) Fit the new learner to the residual
        2) Multiply its parameters by 'shrinkage'
        """
        self.add_weak_learner(i)
        wl = self.learners_for_var[i][-1]
        optimizer = optim.Adam(wl.parameters(), lr=lr)
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            mask_row = W[i, :]
            masked_input = X * mask_row
            pred = wl(masked_input)
            pred = torch.clamp(pred, -1e4, 1e4)
            loss = torch.mean((pred - residual_i)**2)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(wl.parameters(), max_norm=1.0)
            optimizer.step()
            if verbose:
                print(f"     [WL-Fit Var {i} Ep {epoch}] Loss={loss.item():.6f}")

        # Shrink the newly trained weak learner's parameters
        with torch.no_grad():
            for param in wl.parameters():
                param *= shrinkage

class DAGBoostingTrainer:
    """
    DAG-Boost trainer with stability measures:
      - Very small init for W
      - Clamping adjacency after each update
      - Possibly small learning rates and sub-batch
      - Shrinkage factor for each new weak learner
    """
    def __init__(
        self,
        d,
        adjacency_mask=None,
        lr_W=1e-4,
        lambda_h=1.0,
        alpha_init=0.0,
        max_iter=3,
        max_num_weak_learners=2,
        hidden_dim=4,
        tol=1e-4,
        patience=2,
        shrinkage=0.1,
        device=torch.device("cpu")
    ):
        self.d = d
        self.model = FunctionalBoostingModel(
            d=d,
            max_num_weak_learners=max_num_weak_learners,
            hidden_dim=hidden_dim
        ).to(device)

        # Very small init for W
        W_init = 1e-6 * torch.randn(d, d, device=device)
        for i in range(d):
            W_init[i, i] = 0.0
        self.W = nn.Parameter(W_init)

        if adjacency_mask is None:
            adjacency_mask = np.ones((d, d), dtype=np.float32)
            np.fill_diagonal(adjacency_mask, 0.)
        self.adjacency_mask = torch.tensor(adjacency_mask, dtype=torch.float32, device=device)

        self.lambda_h = lambda_h
        self.alpha = alpha_init
        self.lr_W = lr_W
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.shrinkage = shrinkage
        self.device = device

        self.best_loss = float('inf')
        self.no_improv_steps = 0
        self.stop_early = False

    def parameters(self):
        return list(self.model.parameters()) + [self.W]

    def apply_domain_mask_and_no_loops(self):
        apply_mask(self.W, self.adjacency_mask)
        # clamp adjacency to [-0.01, 0.01]
        with torch.no_grad():
            self.W.data = torch.clamp(self.W.data, -0.01, 0.01)

    def augmented_lagrangian_loss(self, X):
        Xhat = self.model(X, self.W)
        recon = squared_loss(X, Xhat)
        h_val = dag_constraint(self.W)
        aug = self.alpha * h_val + 0.5 * self.lambda_h * (h_val ** 2)
        return recon + aug, recon, h_val

    def update_dual(self, h_val):
        self.alpha = self.alpha + self.lambda_h * h_val.item()

    def train(
        self,
        X,
        batch_size=128,
        n_inner_epochs=5,
        fit_new_learner_epochs=5,
        verbose=True
    ):
        """
        Outer loop with sub-batching and partial fits, 
        plus clamp, gradient clip, and shrinkage in each iteration.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        N_full = X.shape[0]

        for outer_iter in range(self.max_iter):
            if verbose:
                print(f"\n===== Outer Iteration {outer_iter+1}/{self.max_iter} =====")

            # Subsample
            indices = np.random.permutation(N_full)
            subset_idx = indices[:batch_size]
            X_sub = X[subset_idx]

            # Residual
            with torch.no_grad():
                Xhat_sub = self.model(X_sub, self.W)
                residuals_sub = torch.clamp(X_sub - Xhat_sub, -1e4, 1e4)
                mse_val = torch.mean(residuals_sub**2).item()

            if verbose:
                print(f"  Sub-batch size={batch_size}, MSE before new learners: {mse_val:.6f}")
            if np.isinf(mse_val) or np.isnan(mse_val):
                print(">>> Residual is inf/nan. Stopping.")
                self.stop_early = True
                break

            # Fit new learners
            d = X_sub.shape[1]
            for i in range(d):
                if self.model.current_counts[i] < self.model.max_num_weak_learners:
                    residual_i_sub = residuals_sub[:, i:i+1]
                    self.model.fit_new_weak_learner(
                        i=i,
                        X=X_sub,
                        residual_i=residual_i_sub,
                        W=self.W,
                        n_epochs=fit_new_learner_epochs,
                        lr=1e-4,
                        shrinkage=self.shrinkage,
                        verbose=(verbose and d <= 20)
                    )

            # Update adjacency
            opt = optim.Adam(self.parameters(), lr=self.lr_W)
            for epoch in range(n_inner_epochs):
                opt.zero_grad()
                loss_total, loss_recon, h_val = self.augmented_lagrangian_loss(X_sub)
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                self.apply_domain_mask_and_no_loops()

            # Update dual
            with torch.no_grad():
                _, _, h_val = self.augmented_lagrangian_loss(X_sub)
            self.update_dual(h_val)

            with torch.no_grad():
                Xhat_sub = torch.clamp(self.model(X_sub, self.W), -1e4, 1e4)
                mse_val = torch.mean((X_sub - Xhat_sub)**2).item()
                h_now = dag_constraint(self.W).item()

            if verbose:
                print(f"  [Iteration {outer_iter+1}] MSE={mse_val:.6f}, h(W)={h_now:.6f}, alpha={self.alpha:.3f}")

            # Early stopping
            if mse_val < self.best_loss - self.tol and not np.isnan(mse_val) and not np.isinf(mse_val):
                self.best_loss = mse_val
                self.no_improv_steps = 0
            else:
                self.no_improv_steps += 1
            if self.no_improv_steps >= self.patience:
                if verbose:
                    print("No improvement; early stopping.")
                self.stop_early = True
                break

        return self.W.detach(), self.model

    def get_binarized_adjacency(self, threshold=0.1):
        return binarize_adjacency(self.W, threshold=threshold)


def run_dagboost_method(
    X_data: np.ndarray,
    threshold=0.1,
    max_iter=3,
    max_num_weak_learners=2,
    hidden_dim=4,
    lambda_h=1.0,
    shrinkage=0.1,
    verbose=False
) -> np.ndarray:
    """
    The stable version of DAG-Boost with inf/nan avoidance:
    - Clamping adjacency
    - Small init
    - Shrinkage for new learners
    - Gradient clipping

    X_data: shape (N, d), input data
    threshold: binarization threshold for adjacency
    max_iter: outer DAG-Boost iterations
    max_num_weak_learners: how many learners per variable
    hidden_dim: MLP dimension
    lambda_h: weight for DAG constraint penalty
    shrinkage: factor to scale newly fitted weak learner parameters
    verbose: whether to print progress

    Returns:
      final_adj: (d, d) binary adjacency
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = X_data.shape[1]
    adjacency_mask = np.ones((d, d), dtype=np.float32)
    np.fill_diagonal(adjacency_mask, 0.0)

    trainer = DAGBoostingTrainer(
        d=d,
        adjacency_mask=adjacency_mask,
        lr_W=1e-4,
        lambda_h=lambda_h,
        alpha_init=0.0,
        max_iter=max_iter,
        max_num_weak_learners=max_num_weak_learners,
        hidden_dim=hidden_dim,
        tol=1e-6,
        patience=2,
        shrinkage=shrinkage,
        device=device
    )
    trainer.train(
        X_data,
        batch_size=128,
        n_inner_epochs=5,
        fit_new_learner_epochs=5,
        verbose=verbose
    )
    W_bin = trainer.get_binarized_adjacency(threshold=threshold)
    G = nx.DiGraph(W_bin)
    remove_all_cycles(G)
    final_adj = nx.to_numpy_array(G, dtype=int)
    return final_adj

def flar(
    X_data: np.ndarray,
    threshold=0.1,
    max_iter=3,
    max_num_weak_learners=2,
    hidden_dim=4,
    lambda_h=1.0,
    shrinkage=0.1,
    verbose=False
) -> np.ndarray:
    """
    A convenience wrapper around `run_dagboost_method`.
    
    Identical arguments and return value, but called 'flar'.
    """
    return run_dagboost_method(
        X_data=X_data,
        threshold=threshold,
        max_iter=max_iter,
        max_num_weak_learners=max_num_weak_learners,
        hidden_dim=hidden_dim,
        lambda_h=lambda_h,
        shrinkage=shrinkage,
        verbose=verbose
    )

###############################################################################
# 3) Demo function
###############################################################################

def main_comparison_stable():
    """
    Example usage demonstration for the "stable" DAG-Boost version.
    """
    # small test scenario
    n_nodes = 5
    n_samples = 100
    true_adj = generate_erdos_renyi_dag(n_nodes, edge_prob=0.3)
    # simulate data (weight_scale=0.01)
    X_data = simulate_sem(true_adj, n_samples, noise_type='gaussian', weight_scale=0.01)

    # run stable DAG-Boost
    est_adj = run_dagboost_method(X_data, verbose=True)

    # Evaluate
    s_shd = shd(true_adj, est_adj)
    s_sid = sid(true_adj, est_adj)
    W_true = true_adj.astype(float)
    W_est  = est_adj.astype(float)
    s_ate_rmse = rmse_ate(W_true, W_est)

    print("\n===== Results =====")
    print(f"SHD: {s_shd}, SID: {s_sid}, ATE_RMSE: {s_ate_rmse:.4f}")


if __name__ == "__main__":
    main_comparison_stable()