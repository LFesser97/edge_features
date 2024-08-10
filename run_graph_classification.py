from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
import torch_geometric.transforms as T

# import custom encodings
from torchvision.transforms import Compose
from custom_encodings import LocalCurvatureProfile, AltLocalCurvatureProfile

from experiments.graph_classification import Experiment


import time
import tqdm
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, digl, borf

import pickle
import wget
import zipfile
import os

def _convert_lrgb(dataset: torch.Tensor) -> torch.Tensor:
    x = dataset[0]
    edge_attr = dataset[1]
    edge_index = dataset[2]
    y = dataset[3]

    return Data(x = x, edge_index = edge_index, y = y, edge_attr = edge_attr)


mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
collab = list(TUDataset(root="data", name="COLLAB"))
reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))

"""
# load peptides dataset from url to the current directory using os and wget
peptides_url = "https://www.dropbox.com/s/ycsq37q8sxs1ou8/peptidesfunc.zip?dl=1"
peptides_zip_filepath = os.getcwd()

wget.download(peptides_url, peptides_zip_filepath)
pepties_zip = os.path.join(peptides_zip_filepath, "peptidesfunc.zip")

with zipfile.ZipFile(pepties_zip, 'r') as zip_ref:
    zip_ref.extractall(peptides_zip_filepath)

peptides_train = torch.load(os.path.join(peptides_zip_filepath, "peptidesfunc", "train.pt"))
peptides_val = torch.load(os.path.join(peptides_zip_filepath, "peptidesfunc", "val.pt"))
peptides_test = torch.load(os.path.join(peptides_zip_filepath, "peptidesfunc", "test.pt"))

peptides = [_convert_lrgb(peptides_train[i]) for i in range(len(peptides_train))] + [_convert_lrgb(peptides_val[i]) for i in range(len(peptides_val))] + [_convert_lrgb(peptides_test[i]) for i in range(len(peptides_test))]
"""

# datasets = {"mutag": mutag, "enzymes": enzymes, "proteins": proteins, "imdb": imdb, "peptides": peptides}
# datasets = {"mutag": mutag, "enzymes": enzymes, "proteins": proteins, "imdb": imdb,
            # "collab": collab, "reddit": reddit}
datasets = {"mutag": mutag, "enzymes": enzymes, "proteins": proteins, "imdb": imdb}

num_vns = 2

for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n,1))

def average_spectral_gap(dataset):
    # computes the average spectral gap out of all graphs in a dataset
    spectral_gaps = []
    for graph in dataset:
        G = to_networkx(graph, to_undirected=True)
        spectral_gap = rewiring.spectral_gap(G)
        spectral_gaps.append(spectral_gap)
    return sum(spectral_gaps) / len(spectral_gaps)

def log_to_file(message, filename="results/graph_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

class SelectiveEncoding:
    """
    An abstract class that contains static methods for selective encoding.    
    """
    @staticmethod
    def select_lcp(graph, dataset_properties):
        """
        Select the encoding method for the graph.
        """
        average_degree = SelectiveRewiring.get_average_degree(graph)
        edge_density = SelectiveRewiring.get_edge_density(graph)
        algebraic_connectivity = SelectiveRewiring.get_algebraic_connectivity(graph)

        """
        if algebraic_connectivity > dataset_properties['algebraic_connectivity'][0]:
            return None
        else:
            return 'lcp'
        """
        if edge_density > dataset_properties['edge_density'][0]:
            return 'lcp'
        else:
            return None
        
    @staticmethod
    def add_zeroes(graph, dim):
        """
        Add zeroes to the graph.
        """
        zeroes = [0 for i in range(graph.num_nodes)]
        padding = torch.tensor([zeroes for i in range(dim)]).T

        if graph.x is not None:
            graph.x = graph.x.view(-1, 1) if graph.x.dim() == 1 else graph.x
            if graph.x.device != padding.device:
                padding = padding.to(graph.x.device)
            graph.x = torch.cat((graph.x, padding), dim=-1)
        else:
            graph.x = torch.cat(padding, dim=-1)
        return graph

class SelectiveRewiring:
    """
    An abstract class that contains static methods for selective rewiring.
    """
    @staticmethod
    def select_borf(graph, dataset_properties):
        """
        Select the rewiring method for the graph.
        """
        average_degree = SelectiveRewiring.get_average_degree(graph)
        edge_density = SelectiveRewiring.get_edge_density(graph)

        if edge_density < dataset_properties['edge_density'][0] and average_degree < dataset_properties['average_degree'][0]:
            return None
        else:
            return 'borf'
        
    @staticmethod
    def select_fosr(graph, dataset_properties):
        """
        Select the rewiring method for the graph.
        """
        average_degree = SelectiveRewiring.get_average_degree(graph)
        edge_density = SelectiveRewiring.get_edge_density(graph)
        algebraic_connectivity = SelectiveRewiring.get_algebraic_connectivity(graph)        

        if algebraic_connectivity < dataset_properties['algebraic_connectivity'][0]:
            return 'fosr'
        else:
            return None
        """
        if edge_density < dataset_properties['edge_density'][0] or average_degree < dataset_properties['average_degree'][0] - dataset_properties['average_degree'][1]:
            if algebraic_connectivity > dataset_properties['algebraic_connectivity'][0]:
                return 'fosr'
            else:
                return None
        else:
            return 'fosr'
        """

    @staticmethod
    def select_rewiring(graph, dataset_properties):
        """
        Select the rewiring method for the graph.
        """
        average_degree = SelectiveRewiring.get_average_degree(graph)
        edge_density = SelectiveRewiring.get_edge_density(graph)
        algebraic_connectivity = SelectiveRewiring.get_algebraic_connectivity(graph)

        if edge_density < dataset_properties['edge_density'][0] and average_degree < dataset_properties['average_degree'][0]:
            return None
        # elif algebraic_connectivity > dataset_properties['algebraic_connectivity'][0]:
            # return 'fosr'
        else:
            return 'borf'

    @staticmethod
    def compute_attributes(dataset):
        """
        Compute the attributes of the dataset.
        """
        dataset_properties = {'average_degree' : [], 'edge_density' : [], 'algebraic_connectivity' : []}

        average_degrees = [SelectiveRewiring.get_average_degree(graph) for graph in dataset]
        edge_densities = [SelectiveRewiring.get_edge_density(graph) for graph in dataset]
        algebraic_connectivities = [SelectiveRewiring.get_algebraic_connectivity(graph) for graph in dataset]

        dataset_properties['average_degree'] = [np.mean(average_degrees), np.std(average_degrees)]
        dataset_properties['edge_density'] = [np.mean(edge_densities), np.std(edge_densities)]
        dataset_properties['algebraic_connectivity'] = [np.mean(algebraic_connectivities), np.std(algebraic_connectivities)]

        return dataset_properties

    @staticmethod
    def get_average_degree(graph):
        """
        Get the average degree of the graph.
        """
        return 2 * graph.num_edges / graph.num_nodes

    @staticmethod
    def get_edge_density(graph):
        """
        Get the edge density of the graph.
        """
        return graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1))

    @staticmethod
    def get_algebraic_connectivity(graph):
        """
        Get the algebraic connectivity of the graph.
        """
        return rewiring.spectral_gap(to_networkx(graph).to_undirected())

default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 4,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 400,
    "eval_every": 1,
    "rewiring": None,
    "num_iterations": 40,
    "patience": 50,
    "output_dim": 2,
    "alpha": 0.1,
    "eps": 0.001,
    "dataset": None,
    "last_layer_fa": False,
    "borf_batch_add" : 20,
    "borf_batch_remove" : 3,
    "sdrf_remove_edges" : False,
    "encoding" : None
})

hyperparams = {
    "mutag": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "proteins": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "imdb": AttrDict({"output_dim": 2}),
    "reddit": AttrDict({"output_dim": 2}),
    "peptides": AttrDict({"output_dim": 10}),
    "pascal": AttrDict({"output_dim": 20}),
    "coco": AttrDict({"output_dim": 80})
}

results = []
args = default_args
args += get_args_from_input()
if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    args += hyperparams[key]
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    energies = []
    print(f"TESTING: {key} ({args.rewiring} - layer {args.layer_type})")

    dataset = datasets[key]
    
    
    # encode the dataset using the given encoding, if args.encoding is not None
    if args.encoding in ["LAPE", "RWPE", "LCP", "LDP", "SUB", "EGO", "VN", "VN-k", "S-LCP"]:

        if os.path.exists(f"data/{key}_{args.encoding}.pt"):
            print('ENCODING ALREADY COMPLETED...')
            dataset = torch.load(f"data/{key}_{args.encoding}.pt")

        elif args.encoding == "LCP":
            print('ENCODING STARTED...')
            lcp = LocalCurvatureProfile()
            for i in range(len(dataset)):
                dataset[i] = lcp.compute_orc(dataset[i])
                print(f"Graph {i} of {len(dataset)} encoded with {args.encoding}")
            torch.save(dataset, f"data/{key}_{args.encoding}.pt")

        elif args.encoding == "VN-k":
            print('ENCODING STARTED...')
            transform = T.VirtualNode()
            for i in range(len(dataset)):
                for j in range(num_vns):
                    dataset[i] = transform(dataset[i])
                print(f"Graph {i} of {len(dataset)} encoded with {args.encoding}")
            torch.save(dataset, f"data/{key}_{args.encoding}_{num_vns}.pt")

        elif args.encoding == "S-LCP":
            print('ENCODING STARTED...')
            lcp = LocalCurvatureProfile()
            dataset_properties = SelectiveRewiring.compute_attributes(dataset)
            for i in range(len(dataset)):
                encoding_method = SelectiveEncoding.select_lcp(dataset[i], dataset_properties)
                if encoding_method == "lcp":
                    dataset[i] = lcp.compute_orc(dataset[i])
                    print(f"Graph {i} of {len(dataset)} encoded with {args.encoding}")
                else:
                    dataset[i] = SelectiveEncoding.add_zeroes(dataset[i], 5)
                    print(f"Graph {i} of {len(dataset)} not encoded")

        else:
            print('ENCODING STARTED...')
            org_dataset_len = len(dataset)
            drop_datasets = []
            current_graph = 0

            for i in range(org_dataset_len):
                if args.encoding == "LAPE":
                    num_nodes = dataset[i].num_nodes
                    eigvecs = np.min([num_nodes, 8]) - 2
                    transform = T.AddLaplacianEigenvectorPE(k=eigvecs)

                elif args.encoding == "RWPE":
                    transform = T.AddRandomWalkPE(walk_length=16)

                elif args.encoding == "LDP":
                    transform = T.LocalDegreeProfile()

                elif args.encoding == "SUB":
                    transform = T.RootedRWSubgraph(walk_length=10)

                elif args.encoding == "EGO":
                    transform = T.RootedEgoNets(num_hops=2)

                elif args.encoding == "VN":
                    transform = T.VirtualNode()

                try:
                    dataset[i] = transform(dataset[i])
                    print(f"Graph {current_graph} of {org_dataset_len} encoded with {args.encoding}")
                    current_graph += 1

                except:
                    print(f"Graph {current_graph} of {org_dataset_len} dropped due to encoding error")
                    drop_datasets.append(i)
                    current_graph += 1

            for i in sorted(drop_datasets, reverse=True):
                dataset.pop(i)

            # save the dataset to a file in the data folder
            torch.save(dataset, f"data/{key}_{args.encoding}.pt")


    print('REWIRING STARTED...')
    start = time.time()
    with tqdm.tqdm(total=len(dataset)) as pbar:
        if args.rewiring == "fosr":
            for i in range(len(dataset)):
                edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=args.num_iterations)
                dataset[i].edge_index = torch.tensor(edge_index)
                dataset[i].edge_type = torch.tensor(edge_type)
                pbar.update(1)
        elif args.rewiring == "sdrf_orc":
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations, remove_edges=False, is_undirected=True, curvature='orc')
                pbar.update(1)
        elif args.rewiring == "sdrf_bfc":
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations, remove_edges=args["sdrf_remove_edges"], 
                        is_undirected=True, curvature='bfc')
                pbar.update(1)
        elif args.rewiring == "borf":
            print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
            print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
            print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = borf.borf3(dataset[i], 
                        loops=args.num_iterations, 
                        remove_edges=False, 
                        is_undirected=True,
                        batch_add=args.borf_batch_add,
                        batch_remove=args.borf_batch_remove,
                        dataset_name=key,
                        graph_index=i)
                pbar.update(1)
        elif args.rewiring == "barf_3":
            print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
            print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
            print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = borf.borf4(dataset[i], 
                        loops=args.num_iterations, 
                        remove_edges=False, 
                        is_undirected=True,
                        batch_add=args.borf_batch_add,
                        batch_remove=args.borf_batch_remove,
                        dataset_name=key,
                        graph_index=i)
                pbar.update(1)
        elif args.rewiring == "barf_4":
            print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
            print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
            print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = borf.borf5(dataset[i], 
                        loops=args.num_iterations, 
                        remove_edges=False,
                        is_undirected=True,
                        batch_add=args.borf_batch_add,
                        batch_remove=args.borf_batch_remove,
                        dataset_name=key,
                        graph_index=i)
                pbar.update(1)
        elif args.rewiring == "digl":
            for i in range(len(dataset)):
                dataset[i].edge_index = digl.rewire(dataset[i], alpha=0.1, eps=0.05)
                m = dataset[i].edge_index.shape[1]
                dataset[i].edge_type = torch.tensor(np.zeros(m, dtype=np.int64))
                pbar.update(1)
        elif args.rewiring == "selective_borf":
            dataset_properties = SelectiveRewiring.compute_attributes(dataset)
            for i in range(len(dataset)):
                rewiring_method = SelectiveRewiring.select_borf(dataset[i], dataset_properties)
                if rewiring_method == "borf":
                    print(f"Graph {i} of {len(dataset)} rewired with BORF")
                    dataset[i].edge_index, dataset[i].edge_type = borf.borf3(dataset[i], 
                        loops=args.num_iterations, 
                        remove_edges=False, 
                        is_undirected=True,
                        batch_add=args.borf_batch_add,
                        batch_remove=args.borf_batch_remove,
                        dataset_name=key,
                        graph_index=i)
                else:
                    print(f"Graph {i} of {len(dataset)} not rewired")
                    dataset[i].edge_index, dataset[i].edge_type = borf.borf3(dataset[i], 
                        loops=0, 
                        remove_edges=False, 
                        is_undirected=True,
                        batch_add=args.borf_batch_add,
                        batch_remove=args.borf_batch_remove,
                        dataset_name=key,
                        graph_index=i)
                pbar.update(1)
        elif args.rewiring == "selective_fosr":
            dataset_properties = SelectiveRewiring.compute_attributes(dataset)
            for i in range(len(dataset)):
                rewiring_method = SelectiveRewiring.select_fosr(dataset[i], dataset_properties)
                if rewiring_method == "fosr":
                    print(f"Graph {i} of {len(dataset)} rewired with FOSR")
                    edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=args.num_iterations)
                    dataset[i].edge_index = torch.tensor(edge_index)
                    dataset[i].edge_type = torch.tensor(edge_type)
                else:
                    print(f"Graph {i} of {len(dataset)} not rewired")
                    edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=0)
                    dataset[i].edge_index = torch.tensor(edge_index)
                    dataset[i].edge_type = torch.tensor(edge_type)
                pbar.update(1)
    end = time.time()
    rewiring_duration = end - start

    print('REWIRING COMPLETED...')

    # create a dictionary of the graphs in the dataset with the key being the graph index
    graph_dict = {}
    for i in range(len(dataset)):
       graph_dict[i] = []
    print('GRAPH DICTIONARY CREATED...') 

    
    #spectral_gap = average_spectral_gap(dataset)
    print('TRAINING STARTED...')
    start = time.time()
    for trial in range(args.num_trials):
        train_acc, validation_acc, test_acc, energy, dictionary = Experiment(args=args, dataset=dataset).run()
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)
        energies.append(energy)
        for name in dictionary.keys():
            if dictionary[name] != -1:
                graph_dict[name].append(dictionary[name])
    end = time.time()
    run_duration = end - start

    # pickle the graph dictionary in a new file depending on the number of layers 
    # if args.num_layers == 4:
        # with open(f"new_results/{key}_{args.layer_type}_{args.encoding}_graph_dict.pickle", "wb") as f:
           # pickle.dump(graph_dict, f)
        # print(f"Graph dictionary for {key} pickled")

    # else:
    
    # if args.encoding == 'VN-k':
        # with open(f"results/{args.num_layers}_layers/{key}_{args.layer_type}_{args.encoding}_{num_vns}_graph_dict.pickle", "wb") as f:
            # pickle.dump(graph_dict, f)
            # print(f"Graph dictionary for {key} pickled")
    if args.rewiring is None:
        with open(f"results/{args.num_layers}_layers/{key}_{args.layer_type}_{args.encoding}_graph_dict.pickle", "wb") as f:
            pickle.dump(graph_dict, f)
        print(f"Graph dictionary for {key} pickled")
    else:
        with open(f"results/{args.num_layers}_layers/{key}_{args.layer_type}_{args.rewiring}_graph_dict.pickle", "wb") as f:
            pickle.dump(graph_dict, f)
        print(f"Graph dictionary for {key} pickled")

    # check if the hidden dimension is not 64
    if args.hidden_dim != 64:
        with open(f"results/{args.num_layers}_layers/{key}_{args.layer_type}_{args.hidden_dim}_hidden_dim_graph_dict.pickle", "wb") as f:
            pickle.dump(graph_dict, f)

    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)
    train_ci = 2 * np.std(train_accuracies)/(args.num_trials ** 0.5)
    val_ci = 2 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
    test_ci = 2 * np.std(test_accuracies)/(args.num_trials ** 0.5)
    energy_ci = 200 * np.std(energies)/(args.num_trials ** 0.5)
    log_to_file(f"RESULTS FOR {key} ({args.rewiring}), {args.num_iterations} ITERATIONS:\n")
    log_to_file(f"average acc: {test_mean}\n")
    log_to_file(f"plus/minus:  {test_ci}\n\n")
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "layer_type": args.layer_type,
        "num_iterations": args.num_iterations,
        "borf_batch_add" : args.borf_batch_add,
        "borf_batch_remove" : args.borf_batch_remove,
        "sdrf_remove_edges" : args.sdrf_remove_edges, 
        "alpha": args.alpha,
        "eps": args.eps,
        "test_mean": test_mean,
        "test_ci": test_ci,
        "val_mean": val_mean,
        "val_ci": val_ci,
        "train_mean": train_mean,
        "train_ci": train_ci,
        "energy_mean": energy_mean,
        "energy_ci": energy_ci,
        "last_layer_fa": args.last_layer_fa,
        "run_duration" : run_duration,
    })

    # Log every time a dataset is completed
    df = pd.DataFrame(results)
    with open(f'results/graph_classification_{args.layer_type}_{args.rewiring}.csv', 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0)