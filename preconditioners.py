import torch 
import ilupp 
import scipy.sparse as spsp
from helpers import get_spdiag
def ic(A):
    n = A.shape[0]

    # Convert PyTorch Sparse Tensor to CSR format
    A_ilupp = A.to_sparse_csr()

    # Extract row pointers (crow_indices), column indices, and values
    rows = A_ilupp.crow_indices().cpu().numpy()
    cols = A_ilupp.col_indices().cpu().numpy()
    values = A_ilupp.values().cpu().numpy()

    
    # Construct SciPy CSR matrix
    A_ilupp = spsp.csr_matrix((values, cols, rows), shape=A.shape, dtype=float)
    
    # Apply ichol Preconditioner
    L = ilupp.ichol0(A_ilupp)

    # Convert SciPy CSR matrix to COO format
    L_coo = L.tocoo()

    # Extract row, column, and value arrays
    L_vals = torch.tensor(L_coo.data, dtype=torch.float64)
    L_rows = torch.tensor(L_coo.row, dtype=torch.float64)
    L_cols = torch.tensor(L_coo.col, dtype=torch.float64)

    # Create PyTorch sparse COO tensor
    return torch.sparse_coo_tensor(torch.stack([L_rows, L_cols]), L_vals, (n, n), dtype=torch.float64)

    
def jacobi(A):
    M_ind  = torch.arange(A.shape[0]).repeat(2,1)
     
    M_vals = get_spdiag(A)
    
    return torch.sparse_coo_tensor(values=M_vals,indices=M_ind,dtype=torch.float64)

