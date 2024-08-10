import torch
import torch.nn as nn
from measure_smoothing import dirichlet_normalized
from torch.nn import ModuleList, Dropout, ReLU, BatchNorm1d, Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GatedGraphConv, GINConv, FiLMConv, global_mean_pool, GATConv, GINEConv, global_add_pool

class RGATConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GATConv(in_features, out_features))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()

        if self.args.last_layer_fa:
            if self.layer_type == "R-GCN" or self.layer_type == "GCN":
                self.last_layer_transform = torch.nn.Linear(self.args.hidden_dim, self.args.output_dim)
            elif self.layer_type == "R-GIN" or self.layer_type == "GIN":
                self.last_layer_transform = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim),nn.BatchNorm1d(self.args.hidden_dim), nn.ReLU(),nn.Linear(self.args.hidden_dim, self.args.output_dim))
            else:
                raise NotImplementedError

    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GAT":
            return RGATConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)

    def forward(self, graph, measure_dirichlet=False):
        x, edge_index, ptr, batch = graph.x, graph.edge_index, graph.ptr, graph.batch
        x = x.float()
        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GAT", "R-GIN", "FiLM"]:
                x_new = layer(x, edge_index, edge_type=graph.edge_type)
            else:
                x_new = layer(x, edge_index)
            if i != self.num_layers - 1:
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)
            if i == self.num_layers - 1 and self.args.last_layer_fa:
                combined_values = global_mean_pool(x, batch)
                combined_values = self.last_layer_transform(combined_values)
                if self.layer_type in ["R-GCN", "R-GIN"]:
                    x_new += combined_values[batch]
                else:
                    x_new = combined_values[batch]
            x = x_new 
        if measure_dirichlet:
            energy = dirichlet_normalized(x.cpu().numpy(), graph.edge_index.cpu().numpy())
            return energy
        x = global_mean_pool(x, batch)
        return x


class GINE(torch.nn.Module):
    def __init__(self, args):
        super(GINE, self).__init__()
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.edge_dim = args.edge_dim
        num_layers = len(list(args.hidden_layers)) + 1
        self.num_relations = 2
        self.hidden_dim = args.hidden_dim
        self.hidden_layers = [self.hidden_dim] * num_layers
        num_features = [self.input_dim] + list(self.hidden_layers) + [self.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=0.5)
        self.act_fn = ReLU()

    def get_layer(self, in_features, out_features):
        return GINEConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)), edge_dim=self.edge_dim)

    def forward(self, graph):
        x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        for i, layer in enumerate(self.layers):
            x_new = layer(x.float(), edge_index, edge_attr.float())
            if i != (self.num_layers - 1):
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)
            x = x_new
        x = global_mean_pool(x.float(), batch)
        return x
    

class MLM(torch.nn.Module):
    """
    A Mix-Layer Model (MLM) with GIN layers
    and a GINE layer at the specified position.
    """
    def __init__(self, args):
        super(MLM, self).__init__()
        self.GINE_pos = args.GINE_pos
        self.input_dim = args.input_dim
        self.edge_dim = args.edge_dim
        self.output_dim = args.output_dim
        num_layers = len(list(args.hidden_layers)) + 1
        self.num_relations = 2
        self.hidden_dim = args.hidden_dim
        self.hidden_layers = [self.hidden_dim] * num_layers
        num_features = [self.input_dim] + list(self.hidden_layers) + [self.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            if i != self.GINE_pos:
                layers.append(self.get_layer(in_features, out_features))
            else:
                layers.append(GINEConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)), edge_dim=self.edge_dim))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=0.5)
        self.act_fn = ReLU()

    def get_layer(self, in_features, out_features):
        return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))

    def forward(self, x, edge_index, edge_attr, batch):
        for i, layer in enumerate(self.layers):
            if i == self.GINE_pos:
                x_new = layer(x.float(), edge_index, edge_attr.float())
                if i != (self.num_layers - 1):
                    x_new = self.act_fn(x_new)
                    x_new = self.dropout(x_new)
                x = x_new
            else:
                x_new = layer(x.float(), edge_index)
                if i != (self.num_layers - 1):
                    x_new = self.act_fn(x_new)
                    x_new = self.dropout(x_new)
                x = x_new 
        x = global_mean_pool(x.float(), batch)
        return x