from leqSolvers import solve_cholesky, solve_diag
import torch 
from time import perf_counter

def cg(AA, bb, precond:dict, atol=1e-6, maxIters=2500):
    x  = torch.zeros(size=torch.Size((AA.shape[0], 1)))
    success = False
    linearSolvers = {'jacobi': solve_diag, 'ic': solve_cholesky, 'learned': solve_cholesky}
    precond_type = precond['type']
    M            = precond['M']
    residuals = []
    total_solve_time = 0
    if precond_type.lower() != 'none':  
        residual = -bb  
        abs_error = torch.norm(residual)
        

        solve = linearSolvers[precond_type]
        
        y = solve(M, residual)
        p = -y 
        
        for iters in  range(maxIters):
            
            beta_denom = torch.dot(residual.flatten(), y.flatten())
            Ap = torch.sparse.mm(AA, p)

            alpha = beta_denom  / (torch.dot(p.flatten(), Ap.flatten()))

            x        += alpha * p 
            residual += alpha * Ap

            residual_norm = torch.linalg.vector_norm(residual).item()
            residuals.append(residual_norm)
            if residual_norm < atol:
                success = True
                break 
            
            
            y = solve(M, residual)
            beta = torch.dot(residual.flatten(), y.flatten()) / beta_denom
            p = -y + beta*p 
        

    else:
        residual = bb 
        p        = residual
        abs_error = torch.norm(residual)
        
        for iters in range(maxIters):
            beta_denom = torch.dot(residual.flatten(), residual.flatten()).item()
        
            Ap  = torch.sparse.mm(AA, p)
            alpha  = beta_denom / (torch.dot(p.flatten(), Ap.flatten())) 
            
            x +=  alpha * p
            
            
            residual -= alpha * Ap 
            
            abs_error = torch.linalg.vector_norm(residual).item()
            residuals.append(abs_error)
            if ((abs_error < atol)):
                success = True
                break
            
            beta = torch.dot(residual.flatten(), residual.flatten()).item() /  beta_denom

            p = residual + beta*p
    if success:
        return x, iters, residuals
    else:
        raise ValueError(f"""Conjugate gradient did not converge to desired 
                        absolute error={atol}. cond(A)={torch.linalg.cond(AA.to_dense())}. precond={precond_type}, det={abs(torch.linalg.det(AA.to_dense()))}
                        """)
        
