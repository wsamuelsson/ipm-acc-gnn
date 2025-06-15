import torch
from helpers import coo_to_scipycsr
import numml.sparse as numsp

def solve_diag(M: torch.sparse_coo_tensor, b: torch.FloatTensor):
    M = M.coalesce()
    values = M.values()
    values = torch.reshape(values, shape=(len(values),1))
    return b / values



def solve_cholesky(L_coo: torch.sparse_coo_tensor, b: torch.Tensor):
    
    L_spsp = coo_to_scipycsr(L_coo)
    L_csr = numsp.SparseCSRTensor(L_spsp)
    U_csr = numsp.SparseCSRTensor(L_spsp.T)

    #Forward substitution (L y = b)
    y = L_csr.solve_triangular(upper=False, unit=False, b=b.flatten())
    
    #Backward substitution (L^T x = y)
    x = U_csr.solve_triangular(upper=True, unit=False, b=y)

    # Convert back to PyTorch tensor
    x = x.view(-1,1)

    return x
    