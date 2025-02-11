"""
dagboost_original.py

The "original" DAG-Boost method for causal discovery, 
using an augmented Lagrangian for acyclicity and 
functional boosting with neural weak learners.

No extra measures for inf/nan prevention.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import time
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

###############################################################################
# 0) Standard data simulation & metrics
###############################################################################

def generate_erdos_renyi_dag(num_nodes, edge_prob):
    """
    Generate an Erdős–Rényi random DAG with num_nodes vertices
    and edge probability edge_prob.
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
    Generate a scale-free DAG using a preferential attachment mechanism.
    """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    if num_nodes < 2:
        return adjacency_matrix

    adjacency_matrix[0, 1] = 1
    in_degrees = np.zeros(num_nodes)
    in_degrees[1] = 1

    for new_node in range(2, num_nodes):
        prob_attachment = (in_degrees[:new_node] + 1) \
                          / np.sum(in_degrees[:new_node] + 1)
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

def simulate_sem(adjacency_matrix, n_samples,
                 noise_type='gaussian',
                 weight_scale=1.0,
                 random_state=None):
    """
    Simulate data from a linear SEM:
        X_j = sum_{i in Pa(j)} W_{i,j} * X_i + noise_j
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
                W[i, j] = np.random.normal(loc=0.0, scale=weight_scale)

    # Generate samples
    X = np.zeros((n_samples, num_nodes))
    for s in range(n_samples):
        for node in topo_order:
            parents = np.where(adjacency_matrix[:, node] == 1)[0]
            parents_sum = np.sum(W[parents, node] * X[s, parents])
            if noise_type.lower() == 'gaussian':
                noise = np.random.normal(loc=0.0, scale=1.0)
            elif noise_type.lower() == 'laplace':
                noise = np.random.laplace(loc=0.0, scale=1.0)
            elif noise_type.lower() == 'exponential':
                noise = np.random.exponential(scale=1.0)
            else:
                raise ValueError("noise_type must be 'gaussian', 'laplace', or 'exponential'.")
            X[s, node] = parents_sum + noise

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
# 1) The "original" DAG-Boost code (no inf/nan prevention)
###############################################################################

def squared_loss(x_true, x_pred):
    return 0.5 * torch.mean((x_true - x_pred)**2)

def dag_constraint(W):
    d = W.shape[0]
    WW = W * W
    expm_WW = torch.matrix_exp(WW)
    h = torch.trace(expm_WW) - d
    return h

def apply_mask(W, mask):
    with torch.no_grad():
        W *= mask
        d = W.shape[0]
        for i in range(d):
            W[i, i] = 0.0

def binarize_adjacency(W, threshold=0.3):
    W_np = W.detach().cpu().numpy()
    W_bin = (abs(W_np) > threshold).astype(float)
    np.fill_diagonal(W_bin, 0.0)
    return W_bin

class WeakLearnerNN(nn.Module):
    """
    A small neural net used as a weak learner for a single variable.
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
    Holds an additive ensemble of weak learners for each variable i.
    """
    def __init__(self, d, max_num_weak_learners=2, hidden_dim=4):
        super().__init__()
        self.d = d
        self.max_num_weak_learners = max_num_weak_learners
        self.hidden_dim = hidden_dim

        # For each var i, a list of (weak_learner).
        self.learners_for_var = [[] for _ in range(d)]
        # Keep them all in one ModuleList so PyTorch sees them
        self.weak_learners = nn.ModuleList()
        self.current_counts = [0]*d

    def forward(self, X, W):
        """
        X: (N, d)
        W: (d, d) real adjacency
        """
        N, d = X.shape
        device = X.device
        Xhat = []
        for i in range(d):
            mask_row = W[i, :]
            masked_input = X * mask_row
            pred_i = torch.zeros((N,1), dtype=X.dtype, device=device)
            # sum over all learners for var i
            for learner in self.learners_for_var[i]:
                pred_i += learner(masked_input)
            Xhat.append(pred_i)
        return torch.cat(Xhat, dim=1)

    def add_weak_learner(self, i):
        wl = WeakLearnerNN(input_dim=self.d, hidden_dim=self.hidden_dim)
        self.weak_learners.append(wl)
        self.learners_for_var[i].append(wl)
        self.current_counts[i] += 1

    def fit_new_weak_learner(self, i, X, residual_i, W,
                             n_epochs=5, lr=0.01, verbose=False):
        """
        Fit a new weak learner for variable i to the current residual.
        """
        self.add_weak_learner(i)
        wl = self.learners_for_var[i][-1]
        optimizer = optim.Adam(wl.parameters(), lr=lr)
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            mask_row = W[i, :]
            masked_input = X * mask_row
            pred = wl(masked_input)
            loss = torch.mean((pred - residual_i)**2)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"    [WL-Fit Var {i} Ep {epoch}] Loss={loss.item():.6f}")

class DAGBoostingTrainer:
    """
    Orchestrates:
    - Real-valued adjacency W with augmented Lagrangian for DAG
    - Functional boosting model with a small number of learners per var
    - Subsampling each outer iteration
    - No special measures for inf/nan
    """
    def __init__(
        self,
        d,
        adjacency_mask=None,
        lr_W=0.01,
        lambda_h=5.0,
        alpha_init=0.0,
        max_iter=3,
        max_num_weak_learners=2,
        hidden_dim=4,
        tol=1e-4,
        patience=2,
        device=torch.device("cpu")
    ):
        self.d = d
        self.model = FunctionalBoostingModel(
            d=d,
            max_num_weak_learners=max_num_weak_learners,
            hidden_dim=hidden_dim
        ).to(device)

        # initialize W
        W_init = 0.01 * torch.randn(d, d, device=device)
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
        self.device = device

        self.best_loss = float('inf')
        self.no_improv_steps = 0
        self.stop_early = False

    def parameters(self):
        return list(self.model.parameters()) + [self.W]

    def apply_domain_mask_and_no_loops(self):
        apply_mask(self.W, self.adjacency_mask)

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
        batch_size=512,
        n_inner_epochs=15,
        fit_new_learner_epochs=5,
        verbose=True
    ):
        """
        Outer loop with sub-batching and partial fits.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        N_full = X.shape[0]

        for outer_iter in range(self.max_iter):
            if verbose:
                print(f"\n===== Outer Iteration {outer_iter+1}/{self.max_iter} =====")

            indices = np.random.permutation(N_full)
            subset_idx = indices[:batch_size]
            X_sub = X[subset_idx]

            # compute residual
            with torch.no_grad():
                Xhat_sub = self.model(X_sub, self.W)
                residuals_sub = X_sub - Xhat_sub
                mse_val = torch.mean((residuals_sub)**2).item()

            if verbose:
                print(f"  Sub-batch size={batch_size}, MSE before new learners: {mse_val:.6f}")

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
                        lr=0.01,
                        verbose=False
                    )

            # update W
            opt = optim.Adam(self.parameters(), lr=self.lr_W)
            for epoch in range(n_inner_epochs):
                opt.zero_grad()
                loss_total, loss_recon, h_val = self.augmented_lagrangian_loss(X_sub)
                loss_total.backward()
                opt.step()
                self.apply_domain_mask_and_no_loops()

            with torch.no_grad():
                _, _, h_val = self.augmented_lagrangian_loss(X_sub)
            self.update_dual(h_val)

            with torch.no_grad():
                Xhat_sub = self.model(X_sub, self.W)
                mse_val = torch.mean((X_sub - Xhat_sub)**2).item()
                h_now = dag_constraint(self.W).item()

            if verbose:
                print(f"  [Iteration {outer_iter+1}] MSE={mse_val:.6f}, h(W)={h_now:.6f}, alpha={self.alpha:.3f}")

            # early stopping
            if mse_val < self.best_loss - self.tol:
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

    def get_binarized_adjacency(self, threshold=0.3):
        """
        Return a thresholded adjacency from W, removing cycles.
        """
        return binarize_adjacency(self.W, threshold=threshold)


def run_dagboost_method(
    X_data: np.ndarray,
    threshold=0.2,
    max_iter=3,
    max_num_weak_learners=2,
    hidden_dim=4,
    lambda_h=5.0,
    verbose=False
) -> np.ndarray:
    """
    The original DAG-Boost approach: 
    no special clamp/shrinkage for inf/nan prevention.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = X_data.shape[1]
    adjacency_mask = np.ones((d, d), dtype=np.float32)
    np.fill_diagonal(adjacency_mask, 0.0)

    trainer = DAGBoostingTrainer(
        d=d,
        adjacency_mask=adjacency_mask,
        lr_W=0.01,
        lambda_h=lambda_h,
        alpha_init=0.0,
        max_iter=max_iter,
        max_num_weak_learners=max_num_weak_learners,
        hidden_dim=hidden_dim,
        tol=1e-5,
        patience=2,
        device=device
    )

    trainer.train(
        X_data,
        batch_size=128,
        n_inner_epochs=10,
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
    threshold=0.2,
    max_iter=3,
    max_num_weak_learners=2,
    hidden_dim=4,
    lambda_h=5.0,
    verbose=False
) -> np.ndarray:
    """
    Wrapper around run_dagboost_method, 
    offering the same interface but called 'flar'.
    """
    return run_dagboost_method(
        X_data=X_data,
        threshold=threshold,
        max_iter=max_iter,
        max_num_weak_learners=max_num_weak_learners,
        hidden_dim=hidden_dim,
        lambda_h=lambda_h,
        verbose=verbose
    )


def main_comparison():
    """
    Example usage demonstration for the "original" DAG-Boost.
    """
    # Just a small test scenario
    n_nodes = 5
    n_samples = 100
    true_adj = generate_erdos_renyi_dag(n_nodes, edge_prob=0.3)
    # Simulate data
    X_data = simulate_sem(true_adj, n_samples, noise_type='gaussian', weight_scale=1.0)

    # Run DAG-Boost
    est_adj = flar(X_data, verbose=True)

    # Evaluate
    s_shd = shd(true_adj, est_adj)
    s_sid = sid(true_adj, est_adj)
    # For ATE, interpret adjacency edges as weight=1
    W_true = true_adj.astype(float)
    W_est = est_adj.astype(float)
    s_ate_rmse = rmse_ate(W_true, W_est)

    print("\n===== Results =====")
    print(f"SHD: {s_shd}, SID: {s_sid}, ATE_RMSE: {s_ate_rmse:.4f}")

if __name__ == "__main__":
    main_comparison()