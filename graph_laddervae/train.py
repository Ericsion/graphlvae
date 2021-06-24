import argparse
import networkx as nx
import os

import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from utils import data
from graph_laddervae.model import GraphVAE
from graphvae.data import GraphAdjSampler
from torch.utils.data import Dataset, DataLoader

CUDA = 0

LR_milestones = [500, 1000]


def build_model(args, max_num_nodes):
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == 'id':
        input_dim = max_num_nodes
    elif args.feature_type == 'deg':
        input_dim = 1
    elif args.feature_type == 'struct':
        input_dim = 2
    model = GraphVAE(input_dim, 64, 256, max_num_nodes)
    return model


def train(args, dataloader, model):
    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    f = open("test_log.txt", 'w+')
    model.train()
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    for epoch in range(5000):
        for batch_idx, data in enumerate(dataloader):
            model.zero_grad()
            print(data['features'].float(), file=f)
            print(data['adj'].float(), file=f)
            features = data['features'].float()
            adj_input = data['adj'].float()

            features = Variable(features).cuda()
            adj_input = Variable(adj_input).cuda()

            loss = model(features, adj_input)
            print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss)
            loss.backward()

            optimizer.step()
            scheduler.step()
            break


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphVAE arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')

    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--max_num_nodes', dest='max_num_nodes', type=int,
                        help='Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')

    parser.set_defaults(dataset='enzymes',
                        feature_type='id',
                        lr=0.001,
                        batch_size=10,
                        num_workers=1,
                        max_num_nodes=41)
    return parser.parse_args()


class MyDataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.features = Data
        self.adj = Label

    # 返回数据集大小
    def __len__(self):
        return len(self.features)

    # 得到数据内容和标签
    def __getitem__(self, index):
        features = torch.Tensor(self.features[index])
        adj = torch.IntTensor(self.adj[index])
        return {'features': features,
                'adj': adj}


def main():
    prog_args = arg_parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
    print('CUDA', CUDA)
    ### running log

    # if prog_args.dataset == 'enzymes':
    #     graphs = data.Graph_load_batch(min_num_nodes=10, name='ENZYMES')
    #     num_graphs_raw = len(graphs)
    # elif prog_args.dataset == 'grid':
    #     graphs = []
    #     for i in range(2, 3):
    #         for j in range(2, 3):
    #             graphs.append(nx.grid_2d_graph(i, j))
    #     num_graphs_raw = len(graphs)
    #
    # if prog_args.max_num_nodes == -1:
    #     max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    # else:
    #     max_num_nodes = prog_args.max_num_nodes
    #     # remove graphs with number of nodes greater than max_num_nodes
    #     graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]
    #
    # graphs_len = len(graphs)
    # print('Number of graphs removed due to upper-limit of number of nodes: ',
    #       num_graphs_raw - graphs_len)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # # graphs_train = graphs[0:int(0.8*graphs_len)]
    # graphs_train = graphs

    # print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    # print('max number node: {}'.format(max_num_nodes))

    # dataset = GraphAdjSampler(graphs_train, max_num_nodes, features=prog_args.feature_type)
    # sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size, 
    #        replacement=False)
    adj = np.load('AdjS.npy')
    features = np.load('HS.npy')
    dataset = MyDataset(features, adj)
    max_num_nodes = 41
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, )
    # model = build_model(prog_args, max_num_nodes).cuda()
    model = GraphVAE(41, 64, 256, max_num_nodes).cuda()
    train(prog_args, dataset_loader, model)


if __name__ == '__main__':
    main()
