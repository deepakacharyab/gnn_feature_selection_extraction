import torch.nn as nn
import torch.nn.functional as F
import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from utils import load_data, load_data_new, accuracy
import time
import argparse
import numpy as np
import time
import argparse
import numpy as np
import torch.optim as optim

import sys
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch

import pickle

# Training settings
parser = argparse.ArgumentParser(description='Cora,citeseer,pubmed')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=0.006, metavar='N',
                    help='learning rate(cora=0.006,citeseer=0.0035,pubmed=0.004)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (cora=400,citeseer=400,pubmed=300)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=12, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N',
                    help='how big is z(cora=64,citeseer=128,pubmed=)')
parser.add_argument('--dataset', type=str, default='citeseer', metavar='N',
                    help='dataset used')
parser.add_argument('--gumbel-hard', type=int, default=0, metavar='N',
                    help='Gumbel Hard used')
# parser.add_argument('--widen-factor', type=int, default=1, metavar='N',
#                     help='how wide is the model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

cuda = args.cuda 
seed  = args.seed
weight_decay = 0
dropout = 0.5

#cuda = not no_cuda and torch.cuda.is_available()

np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Load data
adj, feat_adj, features, labels, idx_train, idx_val, idx_test, adj_dense = load_data_new(args.dataset)
torch.set_printoptions(threshold=10000)


loss_hist = []
acc_hist = []
loss_train_arr = []
acc_train_arr = []
acc_val_arr = []
acc_test_arr = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if(args.dataset == 'cora'):
    temp = 4
    featuresSelected = 225
elif(args.dataset == 'citeseer'):
    temp = 5
    featuresSelected = 450
elif(args.dataset == 'pubmed'):
    temp = 1.75
    featuresSelected = 105

num_feat = len(features[1])
num_hidden = args.hidden_size

A_hat = adj_dense

A_hat_tensor = torch.Tensor(A_hat).to(device)
X_tensor = torch.Tensor(features).to(device)
y_tensor = torch.LongTensor(labels).to(device)    

tr_acc = 0.0
val_acc = 0.0
test_acc = 0.0

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features,flag2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = featuresSelected
            
        if flag2:
            self.WF = Parameter(torch.FloatTensor(in_features,self.k))        
            self.weight = Parameter(torch.FloatTensor(self.k,out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        self.reset_parameters(flag2)

    def reset_parameters(self,x):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if x:
            stdv2 = 1. / math.sqrt(self.WF.size(1))
            self.WF.data.uniform_(-stdv2, stdv2)

    def forward(self, features, adj,flag,temp):
        if flag:
            global featureSelector
            featureSelector = self.WF 
            support = torch.mm(features,F.gumbel_softmax(self.WF,hard=False,tau=temp,dim=0))
            output = torch.mm(adj, support)
            op = torch.mm(output, self.weight)
        else:
            output = torch.mm(adj, features)
            op = torch.mm(output, self.weight)
        
        return op

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
    
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,features):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, True)
        self.gc2 = GraphConvolution(nhid, nclass, False)
        self.dropout = dropout

    def forward(self, x, adj,temp):
        x = self.gc1(x, adj, True, temp)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, False, temp)
        return F.log_softmax(x,dim=1)
    
model = GCN(nfeat=num_feat, nhid=num_hidden, nclass=labels.max().item() + 1,
            dropout=dropout,features=features)


def test():
    global test_acc
    model.eval()
    output2 = model(features, adj,temp) 
    loss_test = F.nll_loss(output2[idx_test], labels[idx_test])
    acc_test = accuracy(output2[idx_test], labels[idx_test])
    #print("Test set results:",
      #    "loss= {:.4f}".format(loss_test.item()),
     #     "accuracy= {:.4f}".format(acc_test.item()))
    if acc_test > test_acc:
        test_acc = acc_test
    acc_test_arr.append(acc_test)
    print('Highest Test Accuracy:->',test_acc)
    

def train(epoch):
    global tr_acc
    global val_acc
    global temp
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    if(args.dataset == 'cora'):
        if(epoch+1) == 25:
            temp = 3.5
        elif(epoch+1) == 50:
            temp = 3
        elif(epoch+1) == 100:
            temp = 2
        elif(epoch+1) == 150:
            temp = 2
        elif(epoch+1) == 200:
            temp = 1.8
    elif(args.dataset == 'citeseer'):
        if(epoch+1) == 25:
            temp = 4.75
        elif(epoch+1) == 50:
            temp = 4.5
        elif(epoch+1) == 125:
            temp = 4.25
        elif(epoch+1) == 150:
            temp = 4
        elif(epoch+1) == 250:
            temp = 4
        elif(epoch+1) == 350:
            temp = 4
    elif(args.dataset == 'pubmed'):
        if(epoch+1) == 25:
            temp = 1.5
        elif(epoch+1) == 50:
            temp = 1.25
        elif(epoch+1) == 100:
            temp = 1.15
        elif(epoch+1) == 150:
            temp = 1.05
        elif(epoch+1) == 200:
            temp = 1
       
    output = model(features, adj, temp)        
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train_arr.append(loss_train)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    if acc_train > tr_acc:
        tr_acc = acc_train
    loss_train.backward()
    optimizer.step()
     
    model.eval()
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_val_arr.append(acc_val)
    acc_train_arr.append(acc_train)
    if acc_val > val_acc:
        val_acc = acc_val
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    print('Training Accuracy:->',tr_acc)
    print('Validation Accuracy:->',val_acc)

    test()
    
   
   
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=0,betas=(0.9, 0.999), eps=1e-08)
for epoch in range(args.epochs):
    train(epoch)

print("Optimization Finished!")  


indices = featureSelector.max(dim=0)
check = torch.topk(indices[0],featuresSelected)[1]

z = []
for i in check:
    z.append(torch.cat([featureSelector[:,i]]))
rank = torch.stack(z).t()

hardRankedMatrix = rank

if args.gumbel_hard == 1:
    hardRankedMatrix = torch.zeros(featureSelector.size())
    for i in range(featureSelector.size()[1]):
        hardRankedMatrix[check[i]][i] = 1
    
    
with open(args.dataset+'.pickle', 'wb') as f:
    pickle.dump(hardRankedMatrix.detach(), f)