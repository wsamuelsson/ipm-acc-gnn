import torch 
from torch_geometric.data import Data 
from torch_geometric.transforms import LocalDegreeProfile, NormalizeFeatures




def update_M(M, AA_new):
    edge_index = AA_new.indices()
    edge_attr = AA_new.values()  # shape [E, 2]
    n = AA_new.size()[0]

    # Compute diagonal dominance ratio
    row, col = edge_index
    vals = AA_new.values().abs()
    diag_mask = row == col

    row_sum = torch.zeros(n).index_add_(0, row, vals)
    diag_vals = torch.zeros(n).index_add_(0, row[diag_mask], vals[diag_mask])
    dd_ratio = diag_vals / (row_sum + 1e-8)  # shape: (n,)

    #Set new features of M
    M.edge_attr=edge_attr
    M.x = None
    M = LocalDegreeProfile()(M)
    x_structural = M.x  # shape [n, d]

    # Add diagonal dominance ratio as first feature column
    dd_ratio = dd_ratio.view(-1, 1)
    x_combined = torch.cat([dd_ratio, x_structural], dim=1)  # shape [n, d+1]
    M.x = x_combined
    #Final normalization
    M = NormalizeFeatures(["x"])(M)
    return M 



def learned(AA):
    edge_index = AA.indices()
    edge_attr = AA.values()  # shape [E, 2]
    n = AA.size()[0]

    # Compute diagonal dominance ratio
    row, col = edge_index
    vals = AA.values().abs()
    diag_mask = row == col

    row_sum = torch.zeros(n).index_add_(0, row, vals)
    diag_vals = torch.zeros(n).index_add_(0, row[diag_mask], vals[diag_mask])
    dd_ratio = diag_vals / (row_sum + 1e-8)  # shape: (n,)

    # Construct initial PyG Data object
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)

    # Apply local structural features
    data = LocalDegreeProfile()(data)
    x_structural = data.x  # shape [n, d]

    # Add diagonal dominance ratio as first feature column
    dd_ratio = dd_ratio.view(-1, 1)
    x_combined = torch.cat([dd_ratio, x_structural], dim=1)  # shape [n, d+1]
    
    n = AA.size(0)
    # Final normalization
    data_out = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        target_edge_index=edge_index, 
        target_edge_attr=edge_attr,
        x=x_combined,
        global_attributes=None, 
        num_nodes=n,
        )
    data_out = NormalizeFeatures(["x"])(data_out)
    return data_out
