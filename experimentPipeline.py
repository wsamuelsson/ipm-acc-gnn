import torch 
import numpy as np 
import pynetgen
from ipm_torch import infeasable_primal_dual_ipm, augment_matrices, sparse_torch_matrix
from dimacsParser import parseDimacs
import os
import random
from helpers import ParamReader
from time import perf_counter
import argparse

N_REPEAT = 1

def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline")
    parser.add_argument(
        '--size',
        type=str,
        choices=['small', 'large'],
        default='small',
        help="Problem size: 'small' or 'large' (default: 'small')'"
    )
    args = parser.parse_args()

    problem_type = args.size
    print(f"Running experiment with problem type: {problem_type}")
    
    file_path = f"params/{problem_type}_params.json"
    p_reader = ParamReader(file_path)
    
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    torch.set_num_threads(8)
    #Set threads and check that is good
    user_threads = os.environ.get("OMP_NUM_THREADS")
    torch_threads = torch.get_num_threads()
    
    assert int(user_threads) == int(torch_threads)

    # Set the seed for reproducibility
    total_seeds = 200
    random.seed(42)
    seed_pool = list(range(1, 1001))
    all_seeds = random.sample(seed_pool, total_seeds)

    # Split into 80% for training and 20% for validation
    training_seeds = all_seeds[:int(total_seeds*0.8)]
    validation_seeds = all_seeds[int(total_seeds*0.8):]
    
    
    path_to_seed_map = {'bigTrainData': training_seeds, 'bigValData':validation_seeds} if problem_type == 'large' else {'minTrainData': training_seeds, 'minValData':validation_seeds}
    #Check that dirs exist or create them
    ...
    total_fail_count = 0
    n_nodes ,density ,n_sources, n_sinks , total_supply, min_cap , max_cap , tsources ,tsinks ,min_cost, max_cost, capacitated, hicost  = p_reader.read_params()            
    file_path = 'net.dimacs'
    cg_iters_train = []
    cg_iters_val = []
    for data_path, seeds in path_to_seed_map.items():
        print(f"Saving to {data_path}..")
        for seed in seeds:
            #Generate network
            try:
                gen = pynetgen.NetgenNetworkGenerator(nodes=n_nodes, density=density, sinks=n_sinks, sources=n_sources, seed=seed, 
                                                supply=total_supply, maxcost=max_cost, mincost=min_cost, maxcap=max_cap,
                                                mincap=min_cap, type=0, tsinks=tsinks, tsources=tsources, capacitated=capacitated, hicost=hicost)
                gen.write(fname=file_path)
            except (IndexError):
                continue
            #Read in the matrices
            c, aeq, beq, aub, bub, arcs, nodes, n_nodes, n_arcs, A_in = parseDimacs(file_path=file_path)
            A, b, c, n_eq, n = augment_matrices(aeq, beq, aub, bub, c)
            A = sparse_torch_matrix(A)
            A = A.coalesce()
            pct_nnz = 100*len(A.values()) / (A.size()[0] * A.size()[1])
            print("% non zeros: ", pct_nnz)        
            #Solve
            t0  = perf_counter()
            pre_cond_timings = [0]*N_REPEAT
            for i in (range(N_REPEAT)):
                
                #Preconditioner: ic for large and jacobi for small 
                preconditioner = 'ic' if problem_type == 'large' else 'jacobi'
                try:
                    x, fun, iters, p_crit, d_crit, pd_crit, avg_precond_time, cg_iters, cg_timings, ipm_timing, = infeasable_primal_dual_ipm(c_in=c, A_in=A, b_in=b, num_eq = n_eq, pre_conditioner_type=preconditioner, save_data=True, data_path=data_path, seed=seed)
                    
                    pre_cond_timings[i] = avg_precond_time
                    t1  = perf_counter()
                    #Take the max of pre conditioning timings
                    max_pct = max(pre_cond_timings)
                    avg_time = (t1 - t0) / N_REPEAT

                    result = f"""time, {avg_time}, iters, {iters}, pct, {max_pct}, n_nodes, {n_nodes}, density, {density}, 
                            n_sinks, {n_sinks}, n_sources, {n_sources}, n_threads, {torch_threads}, preconditioner, {preconditioner}, max_cg_iters, {max(cg_iters)}
                            ,min_cg_iters, {min(cg_iters)}, avg_cg_iters, {np.mean(cg_iters)}, max_cg_time, {max(cg_timings)}
                            ,min_cg_timing, {min(cg_timings)}, avg_cg_timing, {np.mean(cg_timings)}  """.strip()

                    print(result)
                    if data_path == 'minTrainData':
                        cg_iters_train.append(max(cg_iters))
                    else:
                        cg_iters_val.append(max(cg_iters))
                except RuntimeError as e:
                    total_fail_count += 1
                    print(e, f"""We did not converge for n_nodes, {n_nodes}, density, {density}, 
                            n_sinks, {n_sinks}, n_sources, {n_sources}""")
    
    #Plot to see max cg iteration distribution
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.hist(cg_iters_train, bins=20, label="Train", density=True)
    plt.hist(cg_iters_val, bins=20, label="Val", density=True)
    plt.legend()
    plt.show()
    print(100*total_fail_count / (len(training_seeds) + len(validation_seeds)))
if __name__ == "__main__":
    main()
