import torch 
import numpy as np 
from time import perf_counter
from dimacsParser import parseDimacs
import argparse
import warnings
import os
import csv
from precondModel import MPGNN
import pynetgen
import random 
from datetime import datetime
#utils
from helpers import augment_matrices, spmm_diag, sparse_torch_matrix, save_ipm_data, save_cg_residuals, ParamReader
from preconditioners import jacobi, ic 
from cg import cg 
from gnn_utils import learned, update_M


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.")
torch.set_default_dtype(torch.float64)
torch._dynamo.config.verbose = True
torch._dynamo.config.capture_scalar_outputs = True





@torch.inference_mode()
def infeasable_primal_dual_ipm(c_in, A_in, b_in, num_eq, sigma=0.8, ptol=1e-6, dtol=1e-6, pdtol=1e-6, pre_conditioner_type='jacobi', save_data=False, data_path='trainData', seed=0):
    shape = A_in.shape
    n = shape[1]
    
    #Make sure we are not modding
    A = A_in.clone().detach()
    b = np.copy(b_in)
    c = np.copy(c_in)


    def compute_step_size(x, delta_x, s, delta_s):
        mask_x = delta_x < 0
        mask_s = delta_s < 0

        alpha_x = torch.min(-x[mask_x] / delta_x[mask_x]) if torch.any(mask_x) else torch.inf
        alpha_s = torch.min(-s[mask_s] / delta_s[mask_s]) if torch.any(mask_s) else torch.inf

        return 0.99*min(1, alpha_x), 0.99*min(1, alpha_s)
    


    #Wright p.108
    x_new      = torch.tensor(10*np.ones(shape=(shape[1], 1)), device=device)
    s_new      = torch.tensor(5*np.ones(shape=c.shape), device=device)
    lambda_new = torch.tensor(np.zeros(shape=b.shape), device=device)
    
    c = torch.tensor(c)
    b = torch.tensor(b)
    
    mu    = torch.dot(x_new.flatten(), s_new.flatten()) / n 
    iters = 0
    

    p_crit  = 1
    d_crit  = 1
    pd_crit = (torch.dot(x_new.flatten(),  s_new.flatten())) / (n+n*abs(torch.dot(x_new.flatten(),  c.flatten())))
    A         = A.coalesce()
    AT        = A.T.coalesce()

    if pre_conditioner_type == 'learned':
        m = MPGNN(n_node_features=6, n_edge_features=1, layers=3, n_global_features=0, n_hidden=16, attention=False).to(device=device)
        
        m.load_state_dict(torch.load('best_mpgnn.pt', map_location=torch.device('cpu'),  weights_only=True))
        m.eval()

        #Warm start model
        dummy_AA = A@AT 
        M = learned(dummy_AA)
        #Do a forward pass 
        m(M, inference=True)
    
    avg_precond_time = 0
    
    
    if save_data:
        dir_path = data_path    
        if dir_path not in os.listdir():
            os.mkdir(dir_path)
        dir_path += "/"
    
    cg_iters = []
    cg_timings = []
    
    #For timing IPM
    t0 = perf_counter()
    
    

    #We use these inside the for loop, but they are constant so we move them outside
    theta_ind  = torch.arange(n).repeat(2,1)
    invX_ind = torch.arange(n).repeat(2,1)
    
    while(  (p_crit > ptol) or (d_crit > dtol) or  (abs(pd_crit) > pdtol) ):
        

        mu *= sigma

        bb_dual   = torch.sparse.mm(-A.T, lambda_new) - s_new + c
        bb_primal = b - torch.sparse.mm(A, x_new) 
        bb_mu     = spmm_diag(x_new, s_new, mu)
    
        
        theta_vals = (x_new / s_new).flatten()
        theta      = torch.sparse_coo_tensor(indices=theta_ind, values=theta_vals, device=device)    
        
        
        invX     = torch.sparse_coo_tensor(indices=invX_ind, values=(1.0/x_new).flatten(), size=torch.Size((n,n)), device=device)
        
        f = bb_dual - torch.sparse.mm(invX, bb_mu)
        d = bb_primal 
        

        AA = torch.sparse.mm(torch.sparse.mm(A, theta), AT)
        
        gg = torch.sparse.mm(torch.sparse.mm(A, theta), f) + d
        
        t0_precond = perf_counter()
        
        if pre_conditioner_type == 'learned':
            with torch.no_grad():
                #First we update data object: Only edge values in AA have changed
                M = update_M(M, AA)
                L = m(M)
                precond = {'type': pre_conditioner_type, 'M': L}            

        elif pre_conditioner_type == 'ic':
            L  = ic(AA)
            
            precond = {'type': pre_conditioner_type, 'M': ic(AA)}
        
        elif pre_conditioner_type == 'jacobi':
            precond = {'type': pre_conditioner_type, 'M': jacobi(AA)}        

        elif pre_conditioner_type == 'None':
            precond = {'type': pre_conditioner_type, 'M': lambda M: None}            
        else:
            raise NotImplementedError(f"Preconditioner {pre_conditioner_type} is not implemented. \n")
        
        t1_precond = perf_counter()
        avg_precond_time += (t1_precond - t0_precond)
        
        #Solve using conjugate gradient
        try:
            t0_cg = perf_counter()
            delta_lam, _cg, cg_residuals = cg(AA, gg, atol=1e-6, precond=precond)
               
            t1_cg = perf_counter()
            cg_timings.append(t1_cg - t0_cg)
            cg_iters.append(_cg)
        
        
        except ValueError as e:
            raise RuntimeError(f"Conjugate gradient did not converge inside IPM at iteration {iters}\n" )
            
            
        if iters >= 200:
            break
        
        #New Newton directions
        delta_x   = torch.sparse.mm(theta, torch.sparse.mm(AT, delta_lam) - f)
        delta_s   = torch.sparse.mm(invX, bb_mu - s_new*delta_x)
        
        #Compute step sizes
        alpha_p, alpha_s = compute_step_size(x_new, delta_x, s_new, delta_s)
        
        if save_data:
            #Save training and validation data
            save_ipm_data(AA, iters, dir_path, seed, p_crit, d_crit, pd_crit)
        
        

        #Update in primal, dual and slack
        x_new      += alpha_p * delta_x 
        lambda_new += alpha_s * delta_lam
        s_new      += alpha_s * delta_s 
        
        p_crit  = (torch.norm(bb_primal) / (1+torch.norm(b))).item()
        d_crit  = (torch.norm(bb_dual) / (1+torch.norm(c))).item()
        pd_crit = ((torch.dot(x_new.flatten(),  s_new.flatten())) / (n+n*torch.dot(x_new.flatten(),  c.flatten()))).item()
        
        iters += 1
    
    
    #Compute final time 
    total_ipm_time = perf_counter() - t0 

    #Compute avg preconditioner time
    avg_precond_time /= iters
    x_new = torch.round(x_new)
    return x_new, torch.dot(c.flatten(), x_new.flatten()).item(), iters, p_crit, d_crit, pd_crit, avg_precond_time, cg_iters, cg_timings, total_ipm_time





def main(): 
    parser = argparse.ArgumentParser(description="Run experiment pipeline")
    parser.add_argument(
        '--size',
        type=str,
        choices=['small', 'large', 'unseen'],
        default='small',
        help="Problem size: 'small' 'large' or 'unseen' (default: 'small')"
    )
    args = parser.parse_args()

    problem_type = args.size
    if problem_type == 'small':
        file_path = 'params/small_params.json'
    if problem_type == 'large':
        file_path = 'params/large_params.json'
    if problem_type == 'unseen':
        file_path = 'params/unseen_params.json'
    
    p_reader = ParamReader(file_path)
    total_seeds = 200
    random.seed(42)
    seed_pool = list(range(1, 1001))
    all_seeds = random.sample(seed_pool, total_seeds)
    training_seeds = all_seeds[:int(total_seeds*0.8)]
    validation_seeds = all_seeds[int(total_seeds*0.8):]
    test_seeds = [seed for seed in seed_pool 
                  if (seed not in validation_seeds) and 
                     (seed not in training_seeds)]
    
    test_seeds = random.sample(test_seeds, 20)
    
    
    ipm_logfile = "ipm_results.csv"
    if not os.path.exists(ipm_logfile):
        with open(ipm_logfile, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "avg_time", "fun_val", "iters", "precond_time", "n_nodes", "n_arcs",
                "preconditioner", "max_cg_iters", "min_cg_iters", "avg_cg_iters",
                "max_cg_time", "min_cg_time", "avg_cg_time"
            ])
    
    
    n_nodes ,density ,n_sources, n_sinks , total_supply, min_cap , max_cap , tsources ,tsinks ,min_cost, max_cost, capacitated, hicost  = p_reader.read_params()
    
    file_path = 'net.dimacs'
    preconditioners = ['learned', 'ic', 'jacobi']
    fail_count = {precond: 0 for precond in preconditioners}

    for seed in test_seeds:
        
        
        try:
            gen = pynetgen.NetgenNetworkGenerator(nodes=n_nodes, density=density, sinks=n_sinks, sources=n_sources, seed=seed, tsources=tsources, tsinks=tsinks,  
                                                supply=total_supply, maxcost=max_cost, maxcap=max_cap, mincost=min_cost, mincap=min_cap,
                                                type=0, capacitated=capacitated, rng=0, hicost=hicost)
            gen.write(fname=file_path)
        except IndexError:
            continue

        c, aeq, beq, aub, bub, arcs, nodes, n_nodes, n_arcs, A_inc = parseDimacs(file_path=file_path)
        n = n_arcs
        A, b, c, num_eq, num_vars = augment_matrices(aeq, beq, aub, bub, c)
        print(f"Systems are of size {A.shape[0]} x {A.shape[0]}")

        A = sparse_torch_matrix(A)
    
        
        for preconditioner in preconditioners:    
            try:
                
                x, fun, iters, p_crit, d_crit, pd_crit, avg_precond_time, cg_iters, cg_timings, ipm_time = infeasable_primal_dual_ipm(
                    c_in=c, A_in=A, b_in=b, pre_conditioner_type=preconditioner, save_data=False, 
                    num_eq=num_eq, data_path="testData", seed=seed)        
            
                
            except (torch._C._LinAlgError) as e:
                print(e)
                fail_count[preconditioner] += 1
                continue 
            
            result = f"""time, {ipm_time}, fun_val, {fun}, iters, {iters}, pct, {avg_precond_time}, n_nodes, {n_nodes}, density, {n_arcs}, 
                        preconditioner, {preconditioner}, max_cg_iters, {max(cg_iters)}
                        ,min_cg_iters, {min(cg_iters)}, avg_cg_iters, {np.mean(cg_iters)}, max_cg_time, {max(cg_timings)}
                        ,min_cg_timing, {min(cg_timings)}, avg_cg_timing, {np.mean(cg_timings)}"""
            print(result)  
            
            with open(ipm_logfile, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ipm_time, fun, iters, avg_precond_time, n_nodes, n_arcs,
                    str(preconditioner),
                    max(cg_iters), min(cg_iters), np.mean(cg_iters),
                    max(cg_timings), min(cg_timings), np.mean(cg_timings)
                ])
            
            
    print(fail_count)
if __name__ == "__main__":
    main()

