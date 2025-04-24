# UAI 2025 FLAR Submission Documentation

## FLAR Installation & Usage

### 1. Installation

Cloned the repository locally, and then you can install in editable mode:

```bash
git clone https://github.com/kauroy1994/flar.git
cd flar
pip install -e .
```

Either approach will download and install FLAR along with its dependencies.

---

### 2. Verifying Installation

After installation, you should be able to open a Python interpreter and import FLAR without error:

```python
python
>>> import flar
>>> flar.__version__
'0.1.0' 
```

---

### 3. Basic Usage

Assuming you have observational data in a NumPy array `X_data` of shape `(N, d)`:

```python
import flar
import numpy as np

X_data = np.random.randn(500, 10)  # Example: 500 samples, 10 variables

flar.run_flar_small(X_data, max_iter=3,verbose=True)
flar.run_flar_large(X_data, threshold=0.1, max_iter=5,verbose=True)
```

### 4. Additional Options

FLAR supports hyperparameters for controlling:
- Number of outer iterations (`max_iter`)
- Number of weak learners per variable
- Shrinkage factor
- Lagrangian penalty strength
- Clamping range for adjacency
- Learning rates for adjacency and weak learner nets

For example:

```python
adjacency_est = flar.run_flar(
    X_data,
    threshold=0.2,
    max_iter=5,
    max_num_weak_learners=5,
    hidden_dim=8,
    lambda_h=1.0,
    shrinkage=0.2,
    verbose=True
)
```

Refer to the docstrings inside the FLAR source code for more detailed explanations of each parameter.

---

FLAR non-linear Usage example:
```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import time
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

###############################
# 1) Data Simulation Functions
###############################

def simulate_non_linear_sem(adjacency_matrix, n_samples,
                            noise_type='gaussian',
                            weight_scale=1.0,
                            random_state=None,
                            non_linear_fn=np.tanh):
    """
    Simulate data from a non-linear SEM:
        X_j = f( sum_{i in Pa(j)} W_{i,j} * X_i ) + noise_j
    where f is a non-linear function (default: tanh).
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_nodes = adjacency_matrix.shape[0]
    G = nx.DiGraph(adjacency_matrix)
    topo_order = list(nx.topological_sort(G))

    # Generate random weights for the DAG edges
    W = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1:
                W[i, j] = np.random.normal(loc=0.0, scale=weight_scale)

    # Generate samples with non-linear SEM
    X = np.zeros((n_samples, num_nodes))
    for s in range(n_samples):
        for node in topo_order:
            parents = np.where(adjacency_matrix[:, node] == 1)[0]
            # Compute the linear contribution from parent nodes
            parents_sum = np.sum(W[parents, node] * X[s, parents])
            # Apply a non-linear function to the sum
            non_linear_term = non_linear_fn(parents_sum)
            # Add noise
            if noise_type.lower() == 'gaussian':
                noise = np.random.normal(loc=0.0, scale=1.0)
            elif noise_type.lower() == 'laplace':
                noise = np.random.laplace(loc=0.0, scale=1.0)
            elif noise_type.lower() == 'exponential':
                noise = np.random.exponential(scale=1.0)
            else:
                raise ValueError("noise_type must be 'gaussian', 'laplace', or 'exponential'.")
            X[s, node] = non_linear_term + noise

    return X

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

#####################################
# 2) DAG-Boosting (Causal Discovery)
#####################################

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
    Holds an additive ensemble of weak learners for each variable.
    """
    def __init__(self, d, max_num_weak_learners=2, hidden_dim=4):
        super().__init__()
        self.d = d
        self.max_num_weak_learners = max_num_weak_learners
        self.hidden_dim = hidden_dim

        # For each var i, a list of weak learners.
        self.learners_for_var = [[] for _ in range(d)]
        # Keep them in a ModuleList so PyTorch sees them.
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
      - Real-valued adjacency W with augmented Lagrangian for DAG acyclicity,
      - Functional boosting model with neural weak learners,
      - Subsampling each outer iteration.
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

        # Initialize W
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
        return binarize_adjacency(self.W, threshold=threshold)

def run_dagboost_method(
    X_data: np.ndarray,
    threshold=0.3,
    max_iter=3,
    max_num_weak_learners=2,
    hidden_dim=4,
    lambda_h=5.0,
    verbose=False
) -> np.ndarray:
    """
    The original DAG-Boost approach.
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
    return final_adj, trainer

def flar(
    X_data: np.ndarray,
    threshold=0.3,
    max_iter=3,
    max_num_weak_learners=2,
    hidden_dim=4,
    lambda_h=5.0,
    verbose=False
) -> np.ndarray:
    """
    Wrapper around run_dagboost_method.
    """
    final_adj, trainer = run_dagboost_method(
        X_data=X_data,
        threshold=threshold,
        max_iter=max_iter,
        max_num_weak_learners=max_num_weak_learners,
        hidden_dim=hidden_dim,
        lambda_h=lambda_h,
        verbose=verbose
    )
    return final_adj, trainer

#############################################
# 3) Integrated Gradients for Adjacency Estimation
#############################################

def integrated_gradients_for_sample(model, W, x, baseline, steps=50, device=torch.device("cpu")):
    """
    Compute integrated gradients for a single sample x.
    
    Returns a (d, d) tensor where each [j,i] entry is the integrated gradient
    for output node j with respect to input feature i.
    """
    x = x.to(device)
    baseline = baseline.to(device)
    x_diff = x - baseline  # (1, d)
    d = x.shape[1]
    ig_matrix = torch.zeros((d, d), device=device)
    
    alphas = torch.linspace(0, 1, steps, device=device)
    for alpha in alphas:
        x_interp = baseline + alpha * x_diff
        x_interp.requires_grad_(True)
        output = model(x_interp, W)  # shape (1, d)
        grads_list = []
        for j in range(output.shape[1]):
            if x_interp.grad is not None:
                x_interp.grad.zero_()
            model.zero_grad()
            output[0, j].backward(retain_graph=True)
            grads_list.append(x_interp.grad.detach().clone())
        grads_tensor = torch.cat(grads_list, dim=0)  # shape (d, d)
        ig_matrix += grads_tensor * x_diff
    ig_matrix /= steps
    return ig_matrix

def compute_ig_matrix(model, W, X_data, steps=50, device=torch.device("cpu")):
    """
    Compute the averaged integrated gradients matrix over all samples.
    """
    model.eval()
    n_samples, d = X_data.shape
    ig_total = np.zeros((d, d))
    baseline = torch.zeros((1, d), dtype=torch.float32, device=device)
    for i in range(n_samples):
        x_sample = torch.tensor(X_data[i:i+1], dtype=torch.float32, device=device)
        ig_sample = integrated_gradients_for_sample(model, W, x_sample, baseline, steps=steps, device=device)
        ig_total += ig_sample.cpu().numpy()
    ig_avg = ig_total / n_samples
    return ig_avg

def compute_adjacency_from_ig(model, W, X_data, threshold=0.2, steps=50, device=torch.device("cpu")):
    """
    Compute a binary adjacency matrix from the integrated gradients matrix.
    """
    ig_matrix = compute_ig_matrix(model, W, X_data, steps=steps, device=device)
    binary_adj = (np.abs(ig_matrix) > threshold).astype(int)
    np.fill_diagonal(binary_adj, 0)
    G = nx.DiGraph(binary_adj)
    G = remove_all_cycles(G)
    final_adj = nx.to_numpy_array(G, dtype=int)
    return final_adj

#############################################
# 4) Main Comparison: Running the Methods
#############################################

def main_comparison():
    """
    Demonstrate:
      - Training on simulated non-linear SEM data,
      - Estimating the DAG via standard DAG-Boost and via integrated gradients.
    """
    # Simulation settings
    n_nodes = 5
    n_samples = 100
    edge_prob = 0.3

    # Generate a true DAG and simulate non-linear data
    true_adj = generate_erdos_renyi_dag(n_nodes, edge_prob=edge_prob)
    X_data = simulate_non_linear_sem(true_adj, n_samples,
                                     noise_type='gaussian',
                                     weight_scale=1.0,
                                     non_linear_fn=np.tanh)

    # Train DAG-Boost on the simulated data
    # flar returns both the estimated adjacency and the trainer (to access model & W)
    dagboost_adj, trainer = flar(X_data, threshold=0.3, verbose=True)

    # Compute the integrated gradients (IG) estimated adjacency matrix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ig_adj = compute_adjacency_from_ig(trainer.model, trainer.W, X_data,
                                       threshold=0.2, steps=50, device=device)

    print("\n===== True Adjacency Matrix =====")
    print(true_adj)
    print("\n===== DAG-Boost Estimated Adjacency (Post-Cycle Removal) =====")
    print(dagboost_adj)
    print("\n===== Integrated Gradients Estimated Adjacency =====")
    print(ig_adj)

    # Evaluate IG-based estimation against the true graph
    s_shd = shd(true_adj, ig_adj)
    s_sid = sid(true_adj, ig_adj)
    W_true = true_adj.astype(float)
    W_est = ig_adj.astype(float)
    s_ate_rmse = rmse_ate(W_true, W_est)

    print("\n===== Evaluation (Integrated Gradients) =====")
    print(f"SHD: {s_shd}, SID: {s_sid}, ATE_RMSE: {s_ate_rmse:.4f}")

if __name__ == "__main__":
    main_comparison()
```

**Enjoy using FLAR** for large-scale causal modeling! 

**For questions or issues**, please open a GitHub issue in the repository or contact `yzi@email.sc.edu`.
