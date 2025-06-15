#For learning
import torch 
from torch_geometric.data import Data
from torch_geometric.nn import  aggr
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import LocalDegreeProfile, NormalizeFeatures, TwoHop
#For logging
import csv
import os
from datetime import datetime
from time import perf_counter
import matplotlib.pyplot as plt
from torch_scatter import scatter_softmax
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=8
torch.manual_seed(42)
torch.set_num_threads(4)
torch._dynamo.config.verbose = True
default_dtype=torch.float32 

if device == "cpu":
    torch.set_default_dtype(torch.float32)
    default_dtype=torch.float32
if device == "cuda":
    torch.set_default_dtype(torch.float32)
    default_dtype=torch.float32 

print(f"Using device: {device}, with default type: {default_dtype}")



#Standard for plotting 
plt.rcParams.update({
    "text.usetex": True,          # Use LaTeX to render text
    "font.family": "serif",       # Use serif font (default in LaTeX)
    "font.serif": ["Computer Modern"],  # Match LaTeX's default font
    "axes.labelsize": 12,         # Customize as needed
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


class GraphLayer(torch.nn.Module):
    #We follow the outline provided in the thesis of: 
    """In each layer: 
                    - Update      edge features
                    - Aggregate   messages
                    - Update      node features
                    
                    """
    def __init__(self, n_node_features, n_edge_features, only_tril=True, skip_connections=True, n_global_features=0, first=False, last=False, attention=False, n_hidden=16, activation='elu', aggregation_func = 'sum', dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dropout = dropout
        self.skip = skip_connections 
        self.attention = attention
        
        if self.skip:
            extra_edge_features = 1
        else: 
            extra_edge_features = 0
        
        self.edge_block_l = MLP([2*n_node_features + (n_edge_features + extra_edge_features ) + n_global_features,  
                               n_hidden,
                               n_edge_features if first else n_edge_features+extra_edge_features 
                               ], 
                              activation=activation, 
                              dropout=self.dropout
                              )
        self.node_block_l = MLP([(n_edge_features)+n_node_features + n_global_features if first else (n_edge_features + extra_edge_features)+n_node_features + n_global_features, 
                               n_hidden,
                               n_node_features], activation=activation, 
                               dropout=self.dropout)
        
        self.edge_block_u = MLP([2*n_node_features + (n_edge_features + extra_edge_features ) + n_global_features, 
                               n_hidden, 
                               n_edge_features
                               ], 
                              activation=activation, 
                              dropout=self.dropout
                              )
        self.node_block_u = MLP([(n_edge_features)+n_node_features + n_global_features, 
                               n_hidden,
                               n_node_features], activation=activation, 
                               dropout=self.dropout)
        

        if self.attention:
            self.attention_block_l = MLP([2*n_node_features + (n_edge_features + extra_edge_features ), 
                                        n_hidden, 
                                        1], activation=activation)
            
            self.attention_block_u = MLP([2*n_node_features + (n_edge_features + extra_edge_features ), 
                                        n_hidden, 
                                        1], activation=activation)
            
        
        aggregations = {
            'sum': aggr.SumAggregation,
            'mean': aggr.MeanAggregation,
            'max': aggr.MaxAggregation,
            'min': aggr.MinAggregation,
            'mul': aggr.MulAggregation,
            'softmax': aggr.SoftmaxAggregation,
            'powermean': aggr.PowerMeanAggregation,
            'median': aggr.MedianAggregation,
            'var': aggr.VarAggregation,
            'std': aggr.StdAggregation,
            'mlp': aggr.MLPAggregation
        }
        


        try:
            self.aggregate_l = aggregations[aggregation_func]()
            self.aggregate_u = aggregations[aggregation_func]()
        except KeyError:
            raise NotImplementedError(f"Error in GraphLayer: {aggregation_func} not implementated")


    def forward(self, x, edge_attr, edge_index, global_attributes=None, batch=None):
        """ x: node_features
            edge_attr: Edge attributes
            edge_index: Edge index
        
        """
        row, col = edge_index
        
       
        
        
        if global_attributes is None:        
            
            if self.attention:
            
                #Message pass on tril part
                aggregated_data_l = torch.cat((edge_attr, x[row], x[col]), dim=1)

                #Use notation from Vacececk for attention - do attention on tril part
                e_ij = self.attention_block_l(aggregated_data_l)
                alpha_ij = scatter_softmax(e_ij.squeeze(-1), row)



                messages_l = self.edge_block_l(aggregated_data_l)
                weighted_messages_l = messages_l * alpha_ij.unsqueeze(-1)

                aggregated_messages_l = self.aggregate_l(weighted_messages_l, row)
                
                aggregated_data_l = torch.cat((x, aggregated_messages_l), dim=1)
                node_features_l = self.node_block_l(aggregated_data_l)
                
                
                #Switch indices and do message passing on upper triangular part
                # flip row and column indices
                edge_index_u = torch.stack([edge_index[1], edge_index[0]], dim=0)
                row, col = edge_index_u

                #Message pass on triu part
                aggregated_data_u = torch.cat((messages_l, node_features_l[row], node_features_l[col]), dim=1)
                messages_u = self.edge_block_u(aggregated_data_u)
                
                
                
                
                    
                e_ij = self.attention_block_u(aggregated_data_u)
                alpha_ij = scatter_softmax(e_ij.squeeze(-1), row)


                weighted_messages_u = messages_u * alpha_ij.unsqueeze(-1)
                
                aggregated_messages_u = self.aggregate_u(weighted_messages_u, row)
                
                aggregated_data_u = torch.cat((node_features_l, aggregated_messages_u), dim=1)
                node_features_u = self.node_block_u(aggregated_data_u)
                #return updated edge features, updated node features
                return messages_u, node_features_u 
            
            else:
                #Message pass on tril part
                aggregated_data_l = torch.cat((edge_attr, x[row], x[col]), dim=1)

                messages_l = self.edge_block_l(aggregated_data_l)
                
                aggregated_messages_l = self.aggregate_l(messages_l, row)
                
                aggregated_data_l = torch.cat((x, aggregated_messages_l), dim=1)
                node_features_l = self.node_block_l(aggregated_data_l)
                
                
                #Switch indices and do message passing on upper triangular part
                # flip row and column indices
                edge_index_u = torch.stack([edge_index[1], edge_index[0]], dim=0)
                row, col = edge_index_u

                #Message pass on triu part
                aggregated_data_u = torch.cat((messages_l, node_features_l[row], node_features_l[col]), dim=1)
                messages_u = self.edge_block_u(aggregated_data_u)
                
                
                aggregated_messages_u = self.aggregate_u(messages_u, row)
                
                aggregated_data_u = torch.cat((node_features_l, aggregated_messages_u), dim=1)
                node_features_u = self.node_block_u(aggregated_data_u)
                #return updated edge features, updated node features
                return messages_u, node_features_u
        
        
        if global_attributes is not None: 
            raise NotImplementedError("Global Attributes not yet supported")

class MLP(torch.nn.Module):
    """Class for representing the MLPs in the message passing GNN. 
        -We use this firstly to update each edge feature
        -We use this secondly to update node features

        Use explictly in forward method of GraphLayer I guess
    
    """
    
    def __init__(self, width:list, activation="tanh", dropout=0.0, final=False):
        super().__init__()
        #Width contains widths for input, hidden(s) and output layer
        #Thus, it should atleast contain 2 layers
        total_layers = len(width)

        if total_layers < 2:
            raise Exception("Error in MLP: MLP needs atleast two layers.")
        
        layers = torch.nn.ModuleList()
        activations = {
            'relu': torch.nn.ReLU,
            'leaky_relu': torch.nn.LeakyReLU,
            'sigmoid': torch.nn.Sigmoid,
            'tanh': torch.nn.Tanh,
            'elu': torch.nn.ELU,
            'selu': torch.nn.SELU,
            'prelu': torch.nn.PReLU,
            'gelu': torch.nn.GELU,
            'softmax': torch.nn.Softmax,
            'log_softmax': torch.nn.LogSoftmax,
            'hardtanh': torch.nn.Hardtanh,
            'softplus': torch.nn.Softplus,
            'softsign': torch.nn.Softsign,
            'hardsigmoid': torch.nn.Hardsigmoid,
            'silu': torch.nn.SiLU,  # Also known as Swish
            'mish': torch.nn.Mish,
            'threshold': torch.nn.Threshold
            }
        
        for l in range(total_layers - 1):
            #Here we just linearly transform from the previous layer
            #to the next layer. Follow with suitable activation function. 
            #layers[l] = W^T x + b
            #layers[l+1] = activation     
            
            linear =  torch.nn.Linear(width[l], width[l+1], bias=True)
            layers.append( linear )
                
            #Initialization
            torch.nn.init.kaiming_normal_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            
            try:
                if l < len(width) - 2:  #Apply activation on output layer
                    layers.append(activations[activation]())
            
            except KeyError:
                raise NotImplementedError(f"Error in MLP: Activation function {activation} not implemented. \n")
        
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MPGNN(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, n_global_features, layers,  twohop = False, only_tril=True, skip_connections=True, attention=False, n_hidden=16, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.two_hop  = twohop
        self.only_tril = only_tril
        self.graph_layers = torch.nn.ModuleList()
        self.skip = skip_connections
        self.n_global_features = n_global_features
        self.attention = attention 

        for i in range(layers):
            first = bool(i == 0)
            last  = bool(i == (layers - 1))
            if i==0:
                self.graph_layers.append(GraphLayer(n_edge_features=n_edge_features, n_node_features=n_node_features,
                                                 n_global_features=n_global_features ,dropout=0.0, skip_connections=False, first=first, last=last, n_hidden=n_hidden
                                                 , only_tril=self.only_tril, attention=self.attention))
            elif last:
                self.graph_layers.append(GraphLayer(n_edge_features=n_edge_features, n_node_features=n_node_features,
                                                 n_global_features=n_global_features ,dropout=0.0, skip_connections=self.skip, first=first, last=last, n_hidden=n_hidden, 
                                                 only_tril=self.only_tril, attention=self.attention, activation='tanh'))
     
            else:
                self.graph_layers.append(GraphLayer(n_edge_features=n_edge_features, n_node_features=n_node_features,
                                                 n_global_features=n_global_features ,dropout=0.0, skip_connections=self.skip, first=first, last=last, n_hidden=n_hidden,
                                                   only_tril=self.only_tril, attention=self.attention))
     
    def forward(self, data: Data, inference=False):
        
        node_attr = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr.unsqueeze(-1)  # shape [E, 1]

        n = data.num_nodes
        global_attributes = data.global_attributes if self.n_global_features > 0 else None
        batch = data.batch

        if self.only_tril:
            row, col = edge_index
            tril_mask = row >= col
            edge_index = edge_index[:, tril_mask]
            edge_attr = edge_attr[tril_mask]

        row, col = edge_index

        
        if self.n_global_features == 0:
            for i, layer in enumerate(self.graph_layers):
                #Add skip features in all but first layer input embeddings
                if (i != 0) and self.skip:
                    edge_attr = torch.cat([edge_attr, L_values], dim=1)
                
                #In first layer we compute inv(sqrt(D)) @ H @ inv(sqrt(D))
                if i == 0:
                    #Normalize inputs so we have unit diagonal
                    row, col = edge_index
                    self_loop_mask = (row == col).to(device)

                    H_diag = torch.zeros((n,1), device=device)
                    H_diag[row[self_loop_mask]] = edge_attr[self_loop_mask]
                    
                    D_inv_sqrt = 1.0 / torch.sqrt(H_diag + 1e-8)
                    scale = D_inv_sqrt[row] * D_inv_sqrt[col]
                    edge_attr = edge_attr * scale


                    #Make sure to propagate normalized input edge embedding through model
                    if self.skip:
                        L_values = edge_attr.clone()


                #Compute updated edge and node attributs
                edge_attr, node_attr = layer(node_attr, edge_attr, edge_index, batch=batch)
        else:

            for i, layer in enumerate(self.graph_layers):
                
                if (i != 0) and self.skip:
                    edge_attr = torch.cat([edge_attr, L_values], dim=1)
                
                
                edge_attr, node_attr, global_attributes = layer(
                    node_attr, edge_attr, edge_index, global_attributes=global_attributes, batch=batch)

        
        
        #Now squish to get elements in the correct range
        edge_attr = torch.nn.functional.tanh(edge_attr)
        return self.output_transformation(edge_index=edge_index, edge_attr=edge_attr, n=n, D_inv_sqrt=D_inv_sqrt)
    
    def output_transformation(self, edge_index, edge_attr, n, D_inv_sqrt):
        row, col = edge_index
        offdiag_mask = row > col  # strict lower triangle

        offdiag_indices = edge_index[:, offdiag_mask]
        vals_offdiag = edge_attr[offdiag_mask][:, 0]  # shape [E_off]

        vals_squared = vals_offdiag ** 2

        # Sum squared off-diagonal values per row index (i.e., per node i)
        row_sums = torch.zeros(n, device=edge_attr.device)
        row_sums = row_sums.index_add(0, offdiag_indices[0], vals_squared)
        
        
        diag_mask = (row == col)
        diag_indices = row[diag_mask]
        
        #Normalize rows
        diag_vals = torch.ones(size=(n,), device=device)
        diag_vals = torch.sqrt(torch.nn.functional.softplus(diag_vals - row_sums + 1e-3))
        diag_indices = torch.stack([torch.arange(n, device=edge_attr.device)] * 2)
        # === Combine into sparse matrix L ===
        all_indices = torch.cat([offdiag_indices, diag_indices], dim=1)
        all_values = torch.cat([vals_offdiag, diag_vals], dim=0)
        
        if torch.is_inference_mode_enabled():
            #Scale again with sqrt(Diagonal of input)
            D_sqrt = 1.0 / D_inv_sqrt
            L = D_sqrt * torch.sparse_coo_tensor(all_indices, all_values, size=(n, n), dtype=torch.float64)
            
            return L.coalesce()

        else:    
            L = torch.sparse_coo_tensor(all_indices, all_values, size=(n, n))
            L = L.coalesce()
            L1 = torch.norm(all_values, p=1) / len(all_values)
            return L, L1 





class SparseMatrixDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.coo')]

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        data_dict = torch.load(file_path, weights_only=True)

        # Load core matrices and vectors
        AA = data_dict['AA'].coalesce()   # target matrix (sparse)
        AA = AA.to(default_dtype)         #set dtype depending on cpu or gpu 
        #AA_old = data_dict['AA_old'].coalesce().to(device)  # previous iteration matrix
        iters = data_dict['iters']                       # global IPM iteration index
        p_crit = data_dict['p_crit']
        d_crit = data_dict['d_crit']
        pd_crit = data_dict['pd_crit']
        

        n = AA.size(0)
        
        # Build edge index and attributes
        edge_index = AA.indices()
        edge_attr = AA.values() 

        n = AA.size(0)

        # Compute diagonal dominance ratio
        row, col = edge_index
        vals = AA.values().abs()
        diag_mask = row == col
        # Compute diagonal values
        diag_mask = row == col
        diag_vals = torch.zeros(n, device=edge_attr.device).index_add_(0, row[diag_mask], edge_attr[diag_mask])

        # Compute inverse sqrt of diagonal (safe)
        D_inv_sqrt = torch.zeros(n, device=edge_attr.device)
        D_inv_sqrt = 1.0 / torch.sqrt(diag_vals)
        # Normalize edge values: v_ij / sqrt(d_i * d_j)
        normalized_values = edge_attr * (D_inv_sqrt[row] * D_inv_sqrt[col])
        
        row_sum = torch.zeros(n, device=row.device).index_add_(0, row, vals)
        diag_vals = torch.zeros(n, device=row.device).index_add_(0, row[diag_mask], vals[diag_mask])
        dd_ratio = diag_vals / (row_sum + 1e-8)  # shape: (n,)

        # Construct initial PyG Data object
        data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)

        # Apply local structural features
        data = LocalDegreeProfile()(data)
        x_structural = data.x  # shape [n, d]
        
        # Add diagonal dominance ratio as first feature column
        dd_ratio = dd_ratio.view(-1, 1)
        x_combined = torch.cat([dd_ratio, x_structural], dim=1)  # shape [n, d+1]
        
        
        # Final normalization
        data_out = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x_combined,
            target_edge_index=edge_index, 
            target_edge_attr=normalized_values, 
            global_attributes=None, 
            num_nodes=n,
            iters=iters, 
            n=n, 
        )
        
        data_out = NormalizeFeatures(["x"])(data_out)
        
        return data_out



def fro_loss_A(L, A,  mu=0.1):
    L = L.to_dense()
    A = A.to_dense()

    return torch.norm(L@L.T - A, 'fro')




def stochastic_fro_loss(L, A):
    #Hutchinsson trace formula
    z = torch.randn((A.shape[0], 1), device=L.device)
    est = L@(L.T@z) - A@z
    norm = torch.linalg.vector_norm(est, ord=2) # vector norm
    return norm 

### --- Validation Function --- ###
def validate(model, dataloader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch.to(device)
            L, _  = model(batch)
            A = torch.sparse_coo_tensor(indices=batch.target_edge_index, values=batch.target_edge_attr, size=L.size(), requires_grad=False)
            loss_val  = fro_loss_A(L, A)
            val_loss += loss_val.item() 
            
    val_loss /= len(dataloader) * batch_size
    return val_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


### --- Main Function --- ###
def main():
    parser = argparse.ArgumentParser(description="Run model training pipeline")
    parser.add_argument(
        '--size',
        type=str,
        choices=['small', 'large'],
        default='small',
        help="Problem size: 'small' or 'large' (default: 'small')"
    )

    args = parser.parse_args()

    problem_type = args.size

    # Dataset and loaders
    train_ds = SparseMatrixDataset(root_dir="minTrainData" if problem_type == 'small' else 'bigTrainData')
    val_ds = SparseMatrixDataset(root_dir="minValData" if problem_type == 'small' else 'bigTrainData')  # Add your validation data dir
    
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=lambda batch: Batch.from_data_list(batch), drop_last=True, num_workers=4)
    val_loader   = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, pin_memory=True,  collate_fn=lambda batch: Batch.from_data_list(batch), drop_last=True, num_workers=4)
    

    model = MPGNN(n_node_features=6, n_edge_features=1, layers=3, n_global_features=0, twohop=False, only_tril=True, n_hidden=16, attention=False).to(device=device)
    if problem_type == 'large':
    #Warm start larger model
        try:
            model.load_state_dict(torch.load('best_mpgnn.pt', weights_only=True))
        except RuntimeError:
            print("MPGNN size has changed. ")
    print(f"# of model params: {count_parameters(model)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    torch.autograd.set_detect_anomaly(True)
    

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    logfile = "training_log.csv"
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create file and write header if it doesn't exist
    if not os.path.exists(logfile):
        with open(logfile, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Start Time", "Epoch", "Train Loss", "Val Loss", "Epoch Duration (s)"])
    
    num_total_epochs = 75 if problem_type == 'small' else 45

    for epoch in range(num_total_epochs):
        model.train()
        total_loss = 0
        t_epoch = perf_counter()
        for i, batch in enumerate(train_loader):
            batch.to(device)
            loss_val = 0
            optimizer.zero_grad() 
            L, L1 = model(batch)  
             
            A = torch.sparse_coo_tensor(indices=batch.target_edge_index, values=batch.target_edge_attr, size=L.size(), requires_grad=False)
            loss_val  += stochastic_fro_loss(L, A)
            
            loss_val.backward()
            optimizer.step()
            total_loss += loss_val.item()

        scheduler.step()
        avg_train_loss = total_loss / (len(train_loader)*batch_size)
        train_losses.append(avg_train_loss)

        # Validation
        val_loss = validate(model, val_loader)
        val_losses.append(val_loss)

        epoch_duration = perf_counter() - t_epoch
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}. Current learning rate: {optimizer.param_groups[0]['lr']}. Took: {epoch_duration:.6f}")

        with open(logfile, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([start_time, epoch + 1, avg_train_loss, val_loss, epoch_duration])
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_mpgnn.pt")
            print(f"Saved model at epoch {epoch+1} with val loss {val_loss:.6f}")
            
    
    # Plotting
    print(L)
    plt.figure()
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    import time
    plt.savefig(f"trainAndValLoss_{time.time_ns()}.pdf")
    plt.show()
if __name__ == "__main__":
    main()
