#!/usr/bin/env python
"""
Comprehensive Comparison of FLAR vs. CAM

This script simulates SEM data under different configurations:
  - Number of nodes: 20, 50, 100
  - DAG types: Erdős–Rényi (ER) and Scale‑Free (SF)
  - SEM types: linear and non‑linear
  - Noise: gaussian, exponential, and laplace

For each configuration, the script:
  1. Generates a true DAG (ER or SF).
  2. Simulates data using either a linear or non‑linear SEM (with a fixed random seed).
  3. Runs your DAG‑Boosting (FLAR) method to learn a continuous weight matrix W,
     computes the total effect matrix T = (I - W)⁻¹ - I, and extracts the estimated ATE from T[0, n_vars-1].
     It also thresholds W to recover a binary graph and computes SHD and RMSE (ATE_RMSE) versus the true structure.
  4. Runs the CAM method from the cdt package (which requires Rscript) on the same data.
  5. Records runtime, SHD, and ATE_RMSE for both methods.
  6. Saves the results as CSV and JSON files.

Dependencies:
    pip install numpy pandas torch networkx statsmodels cdt
    Ensure that cdt.SETTINGS.rpath is set to your Rscript path.

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import time
import statsmodels.api as sm

# -------------------------------
# 1. Data Simulation and DAG Generation
# -------------------------------

def generate_erdos_renyi_dag(num_nodes, edge_prob):
    perm = np.random.permutation(num_nodes)
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.rand() < edge_prob:
                adj[perm[i], perm[j]] = 1
    return adj

def generate_scale_free_dag(num_nodes):
    G = nx.scale_free_graph(num_nodes, seed=42)
    G_simple = nx.DiGraph(G)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    G_dag = remove_all_cycles(G_simple.copy())
    adj = nx.to_numpy_array(G_dag, dtype=int)
    np.fill_diagonal(adj, 0)
    return adj

def remove_all_cycles(G):
    while True:
        try:
            cycle = nx.find_cycle(G, orientation='original')
            for (u, v, _) in cycle:
                G.remove_edge(u, v)
        except nx.NetworkXNoCycle:
            break
    return G

def simulate_sem_with_weights(adjacency_matrix, n_samples, noise_type='gaussian', weight_scale=1.0, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    d = adjacency_matrix.shape[0]
    G = nx.DiGraph(adjacency_matrix)
    topo_order = list(nx.topological_sort(G))
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if adjacency_matrix[i, j] == 1:
                W[i, j] = np.random.normal(loc=0.0, scale=weight_scale)
    X = np.zeros((n_samples, d))
    for s in range(n_samples):
        for node in topo_order:
            parents = np.where(adjacency_matrix[:, node] == 1)[0]
            val = np.sum(W[parents, node] * X[s, parents])
            if noise_type.lower() == 'gaussian':
                noise = np.random.normal(scale=1.0)
            elif noise_type.lower() == 'exponential':
                noise = np.random.exponential(scale=1.0)
            elif noise_type.lower() in ['laplace', 'laplacian']:
                noise = np.random.laplace(scale=1.0)
            else:
                raise ValueError("noise_type must be gaussian, exponential, or laplace")
            X[s, node] = val + noise
    return X, W

def simulate_non_linear_sem_with_weights(adjacency_matrix, n_samples, noise_type='gaussian', 
                                           weight_scale=1.0, random_state=42, non_linear_fn=np.tanh):
    if random_state is not None:
        np.random.seed(random_state)
    d = adjacency_matrix.shape[0]
    G = nx.DiGraph(adjacency_matrix)
    topo_order = list(nx.topological_sort(G))
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if adjacency_matrix[i, j] == 1:
                W[i, j] = np.random.normal(loc=0.0, scale=weight_scale)
    X = np.zeros((n_samples, d))
    for s in range(n_samples):
        for node in topo_order:
            parents = np.where(adjacency_matrix[:, node] == 1)[0]
            val = np.sum(W[parents, node] * X[s, parents])
            nl_val = non_linear_fn(val)
            if noise_type.lower() == 'gaussian':
                noise = np.random.normal(scale=1.0)
            elif noise_type.lower() == 'exponential':
                noise = np.random.exponential(scale=1.0)
            elif noise_type.lower() in ['laplace', 'laplacian']:
                noise = np.random.laplace(scale=1.0)
            else:
                raise ValueError("noise_type must be gaussian, exponential, or laplace")
            X[s, node] = nl_val + noise
    return X, W

# -------------------------------
# 2. Evaluation Metrics
# -------------------------------

def shd(true_adj, est_adj):
    return int(np.sum(true_adj != est_adj))

def compute_total_effect_matrix(W):
    d = W.shape[0]
    I = np.eye(d)
    try:
        inv = np.linalg.inv(I - W)
    except np.linalg.LinAlgError:
        inv = I
    return inv - I

def rmse_ate(W_true, W_est):
    T_true = compute_total_effect_matrix(W_true)
    T_est = compute_total_effect_matrix(W_est)
    return np.sqrt(np.mean((T_true - T_est)**2))

def threshold_W(W, thr=0.3):
    W_bin = (np.abs(W) > thr).astype(int)
    np.fill_diagonal(W_bin, 0)
    return W_bin

# -------------------------------
# 3. FLAR Implementation (DAG‑Boosting)
# -------------------------------

def squared_loss(x_true, x_pred):
    return 0.5 * torch.mean((x_true - x_pred)**2)

def dag_constraint(W):
    d = W.shape[0]
    WW = W * W
    expm_WW = torch.matrix_exp(WW)
    return torch.trace(expm_WW) - d

def apply_mask(W, mask):
    with torch.no_grad():
        W *= mask
        d = W.shape[0]
        for i in range(d):
            W[i, i] = 0.0

def binarize_adjacency(W, threshold=0.3):
    W_np = W.detach().cpu().numpy()
    W_bin = (np.abs(W_np) > threshold).astype(float)
    np.fill_diagonal(W_bin, 0)
    return W_bin

class WeakLearnerNN(nn.Module):
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
        preds = []
        for i in range(d):
            mask = W[i, :]
            masked = X * mask
            pred = torch.zeros((N, 1), dtype=X.dtype, device=device)
            for learner in self.learners_for_var[i]:
                pred += learner(masked)
            preds.append(pred)
        return torch.cat(preds, dim=1)
    def add_weak_learner(self, i):
        wl = WeakLearnerNN(input_dim=self.d, hidden_dim=self.hidden_dim)
        self.weak_learners.append(wl)
        self.learners_for_var[i].append(wl)
        self.current_counts[i] += 1
    def fit_new_weak_learner(self, i, X, residual, W, n_epochs=5, lr=0.01, verbose=False):
        self.add_weak_learner(i)
        wl = self.learners_for_var[i][-1]
        optimizer = optim.Adam(wl.parameters(), lr=lr)
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            mask = W[i, :]
            masked = X * mask
            pred = wl(masked)
            loss = torch.mean((pred - residual)**2)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"    [Var {i} Epoch {epoch}] Loss={loss.item():.6f}")

class DAGBoostingTrainer:
    def __init__(self, d, adjacency_mask=None, lr_W=0.01, lambda_h=5.0, alpha_init=0.0,
                 max_iter=3, max_num_weak_learners=2, hidden_dim=4, tol=1e-4,
                 patience=2, device=torch.device("cpu")):
        self.d = d
        self.model = FunctionalBoostingModel(d, max_num_weak_learners, hidden_dim).to(device)
        W_init = 0.01 * torch.randn(d, d, device=device)
        for i in range(d):
            W_init[i, i] = 0.0
        self.W = nn.Parameter(W_init)
        if adjacency_mask is None:
            adjacency_mask = np.ones((d, d), dtype=np.float32)
            np.fill_diagonal(adjacency_mask, 0)
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
        rec_loss = squared_loss(X, Xhat)
        h = dag_constraint(self.W)
        aug = self.alpha * h + 0.5 * self.lambda_h * (h**2)
        return rec_loss + aug, rec_loss, h
    def update_dual(self, h):
        self.alpha = self.alpha + self.lambda_h * h.item()
    def train(self, X, batch_size=128, n_inner_epochs=10, fit_new_learner_epochs=5, verbose=True):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)
        N = X.shape[0]
        for outer in range(self.max_iter):
            indices = np.random.permutation(N)
            batch_idx = indices[:batch_size]
            X_batch = X[batch_idx]
            with torch.no_grad():
                Xhat = self.model(X_batch, self.W)
                residual = X_batch - Xhat
                mse = torch.mean(residual**2).item()
            if verbose:
                print(f"Outer Iteration {outer+1}/{self.max_iter}, Batch MSE: {mse:.6f}")
            for i in range(self.d):
                if self.model.current_counts[i] < self.model.max_num_weak_learners:
                    res = residual[:, i:i+1]
                    self.model.fit_new_weak_learner(i, X_batch, res, self.W,
                                                     n_epochs=fit_new_learner_epochs,
                                                     lr=0.01, verbose=False)
            opt = optim.Adam(self.parameters(), lr=self.lr_W)
            for inner in range(n_inner_epochs):
                opt.zero_grad()
                loss_total, loss_rec, h = self.augmented_lagrangian_loss(X_batch)
                loss_total.backward()
                opt.step()
                self.apply_domain_mask_and_no_loops()
            with torch.no_grad():
                _, _, h = self.augmented_lagrangian_loss(X_batch)
            self.update_dual(h)
        return self.W.detach(), self.model

def run_dagboost_method(X_data, threshold=0.3, max_iter=3, max_num_weak_learners=2,
                        hidden_dim=4, lambda_h=5.0, verbose=False):
    device = torch.device("cpu")
    d = X_data.shape[1]
    adj_mask = np.ones((d, d), dtype=np.float32)
    np.fill_diagonal(adj_mask, 0)
    trainer = DAGBoostingTrainer(d, adjacency_mask=adj_mask, lr_W=0.01, lambda_h=lambda_h,
                                 alpha_init=0.0, max_iter=max_iter,
                                 max_num_weak_learners=max_num_weak_learners,
                                 hidden_dim=hidden_dim, tol=1e-5, patience=2,
                                 device=device)
    trainer.train(X_data, batch_size=128, n_inner_epochs=10, fit_new_learner_epochs=5, verbose=verbose)
    return trainer.W.detach().cpu().numpy(), trainer

def flar(X_data, threshold=0.3, max_iter=3, max_num_weak_learners=2, hidden_dim=4,
         lambda_h=5.0, verbose=False):
    return run_dagboost_method(X_data, threshold=threshold, max_iter=max_iter,
                               max_num_weak_learners=max_num_weak_learners,
                               hidden_dim=hidden_dim, lambda_h=lambda_h, verbose=verbose)

# -------------------------------
# 4. CAM Method from cdt (Baseline)
# -------------------------------
import cdt
cdt.SETTINGS.rpath = "/usr/bin/Rscript"  # Update this path if necessary
from cdt.causality.graph import CAM

def run_cam(X_data):
    # Convert simulated data to DataFrame
    df = pd.DataFrame(X_data)
    cam_model = CAM()
    output_graph = cam_model.predict(df)
    A_cam = nx.to_numpy_array(output_graph, dtype=int)
    # Remove cycles if any
    G_cam = nx.DiGraph(A_cam)
    remove_all_cycles(G_cam)
    return nx.to_numpy_array(G_cam, dtype=int)

# -------------------------------
# 5. Main Comprehensive Experiment: FLAR vs. CAM
# -------------------------------

if __name__ == '__main__':
    run_cell = True  # Set to True to run the experiment
    if not run_cell:
        raise Exception("Execution halted.")
    
    # Experimental settings
    node_settings = [20, 50, 100]   # numbers of nodes to test
    dag_types = ['ER', 'SF']        # DAG generation: Erdos-Renyi and Scale-Free
    sem_types = ['linear', 'non-linear']
    noise_types = ['gaussian', 'exponential', 'laplace']
    n_samples = 500                 # sample size
    weight_scale = 1.0
    
    results = []
    for n_vars in node_settings:
        edge_prob = 2.0 / (n_vars - 1)  # ER setting: expected average degree ~2
        for dag in dag_types:
            if dag == 'ER':
                true_adj = generate_erdos_renyi_dag(n_vars, edge_prob)
            elif dag == 'SF':
                true_adj = generate_scale_free_dag(n_vars)
            else:
                continue
            for sem in sem_types:
                for noise in noise_types:
                    config = f"n_vars={n_vars}, DAG={dag}, SEM={sem}, Noise={noise}"
                    print("\n[Config]", config)
                    # Simulate data
                    if sem == 'linear':
                        X_data, W_true = simulate_sem_with_weights(true_adj, n_samples,
                                                                   noise_type=noise,
                                                                   weight_scale=weight_scale,
                                                                   random_state=42)
                    else:
                        X_data, W_true = simulate_non_linear_sem_with_weights(true_adj, n_samples,
                                                                              noise_type=noise,
                                                                              weight_scale=weight_scale,
                                                                              random_state=42,
                                                                              non_linear_fn=np.tanh)
                    
                    #normalize the data
                    X_data = (X_data - X_data.mean(axis=0)) / (X_data.std(axis=0) + 1e-8)
                    
                    # Run FLAR
                    t0 = time.time()
                    W_fl, _ = flar(X_data, threshold=0.3, max_iter=5,
                                   max_num_weak_learners=50, hidden_dim=40,
                                   lambda_h=5.0, verbose=False)
                    runtime_fl = time.time() - t0
                    T_fl = compute_total_effect_matrix(W_fl)
                    fl_ate = T_fl[0, n_vars-1]   # Assume treatment = node0, outcome = node n_vars-1
                    est_adj_fl = threshold_W(W_fl, thr=0.3)
                    shd_fl = shd(true_adj, est_adj_fl)
                    ate_rmse_fl = rmse_ate(W_true, est_adj_fl.astype(float))
                    
                    results.append({
                        'Method': 'FLAR',
                        'n_vars': n_vars,
                        'DAG_type': dag,
                        'SEM_type': sem,
                        'Noise': noise,
                        'Runtime_sec': runtime_fl,
                        'SHD': shd_fl,
                        'ATE_RMSE': ate_rmse_fl,
                        'FLAR_ATE': fl_ate
                    })
                    
                    # Run CAM method
                    t0 = time.time()
                    A_cam = run_cam(X_data)
                    runtime_cam = time.time() - t0
                    shd_cam = shd(true_adj, A_cam)
                    ate_rmse_cam = rmse_ate(W_true, A_cam.astype(float))
                    
                    results.append({
                        'Method': 'CAM',
                        'n_vars': n_vars,
                        'DAG_type': dag,
                        'SEM_type': sem,
                        'Noise': noise,
                        'Runtime_sec': runtime_cam,
                        'SHD': shd_cam,
                        'ATE_RMSE': ate_rmse_cam,
                        'FLAR_ATE': np.nan   # Not applicable for CAM
                    })
    
    df_results = pd.DataFrame(results)
    print("\n===== Comprehensive Comparison Results (FLAR vs. CAM) =====")
    print(df_results)
    
    # Save the dataframe to CSV and JSON
    df_results.to_csv("comparison_flar_cam_results.csv", index=False)
    df_results.to_json("comparison_flar_cam_results.json", orient="records", lines=True)
    print("Results have been saved to 'comparison_flar_cam_results.csv' and 'comparison_flar_cam_results.json'.")