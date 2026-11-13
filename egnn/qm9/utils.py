import torch

def compute_mean_mad(dataloaders, label_property):
    dataset = dataloaders['train'].dataset
    values = dataset.data[label_property]
    if hasattr(dataset, 'indices'):
        values = values[dataset.indices]
    meann = torch.mean(values)
    mad = torch.mean(torch.abs(values - meann))
    return meann, mad

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    cache_key = (n_nodes, str(device))
    if cache_key in edges_dic:
        edges_dic_b = edges_dic[cache_key]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[cache_key] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    edges_dic[cache_key][batch_size] = edges
    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars
