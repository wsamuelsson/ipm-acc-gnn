import numpy as np
import scipy.sparse as spsp
import torch 
import time 
import json 

def augment_matrices(aeq, beq, aub, bub, c):
    # Convert inputs to proper shapes
    if beq is not None and beq.size > 0:
        beq = beq.reshape((-1, 1))
    else:
        beq = np.empty((0, 1))  # Empty column vector

    if bub is not None and bub.size > 0:
        bub = bub.reshape((-1, 1))
    else:
        bub = np.empty((0, 1))  # Empty column vector

    # Determine problem sizes
    num_slack = aub.shape[0] if aub is not None and aub.size > 0 else 0
    num_vars = c.size  # Original decision variable count
    num_eq = aeq.shape[0]

    # Construct A matrix
    if aeq is not None and aeq.shape[0] > 0:
        A = spsp.vstack((aeq, aub)) if aub is not None and aub.shape[0] > 0 else aeq
    else:
        A = aub if aub is not None and aub.shape[0] > 0 else spsp.csr_matrix((0, num_vars))
    
    
    # Add slack variables for inequality constraints
    if num_slack > 0:
        slack_matrix = spsp.vstack((spsp.csr_matrix((A.shape[0] - num_slack, num_slack)), spsp.eye(num_slack, format='csr')))
        A = spsp.hstack((A, slack_matrix))
    # Construct b vector
    b = np.vstack((beq, bub)) if bub.size > 0 else beq

    # Construct c vector (adding zeros for slack variables)
    c = np.vstack((c.reshape(-1, 1), np.zeros((num_slack, 1)))) if num_slack > 0 else c.reshape(-1, 1)

    return A, b, c, num_eq, num_vars

def sparse_diagonal_coo(diag: torch.FloatTensor):
    values = diag.flatten()
    indices = torch.tensor([[i, i] for i in range(len(values))], dtype=torch.long).T
    X = torch.sparse_coo_tensor(indices=indices, values=values)
    return X

def spmm_diag(x:torch.tensor, s:torch.tensor, mu:float):
    """Returns the vector cs such that cs_i = -x_i * c_i + mu"""
    cs = (-1.0*x) * s + mu  
    return cs 


def get_spdiag(A):
    values = A.values()
    ind = A.indices()
    mask = (ind[0] == ind[1])
    return values[mask] 

def coo_to_scipycsr(A: torch.sparse_coo_tensor):
    #From https://github.com/paulhausner/neural-incomplete-factorization/blob/main/neuralif/utils.py
    A = A.coalesce()
    d = A.values().squeeze().numpy()
    i, j = A.indices().numpy()
    A_s = spsp.coo_matrix((d, (i, j)))
    
    return A_s.tocsr()


def sparse_torch_matrix(A: np.array):
    A = spsp.coo_matrix(A)
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.tensor(values)
    shape = A.shape
    
    A = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return A


def save_ipm_data(AA, iters, dir_path, seed, p_crit, d_crit, pd_crit):
    row_dim = AA.size()[0]
    col_dim = row_dim
    epoch_time_ms = int(time.time() * 1000)  # Milliseconds
    file_path = dir_path + f"leqs_{row_dim}x{col_dim}_{seed}_{iters}_{epoch_time_ms}.coo"    
    torch.save({'AA':AA, 'iters':iters, 
                        'p_crit':p_crit, 'd_crit':d_crit, 'pd_crit':pd_crit}, file_path)

def save_cg_residuals(cg_residuals, pre_conditioner_type, row_dim, col_dim, seed, iters, dir_path):
    epoch_time_ms = int(time.time() * 1000)  # Milliseconds
    with open(dir_path + "cg_residuals/"+ f"{pre_conditioner_type}_residuals_{row_dim}x{col_dim}_{seed}_{iters}_{epoch_time_ms}.txt", 'w') as outfile:
                    outfile.write('\n'.join(str(i) for i in cg_residuals))

class ParamReader():
    def __init__(self, f):
         self.file_name = f 
    def read_params(self) -> tuple:
        with open(self.file_name, 'r') as f:
            params = json.load(f)
        return params['n_nodes'], params['density'], params['n_sources'], params['n_sinks'], params['total_supply'], params['min_cap'], params['max_cap'], params['tsources'], params['tsinks'], params['min_cost'], params['max_cost'], params['capacitated'], params['hicost']
            

