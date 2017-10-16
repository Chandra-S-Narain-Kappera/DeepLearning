from __future__ import print_function
import argparse
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from visualization import plot_kernels,plot_activations
#######################################################################################
# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='M',
                    help='SGD|Adam')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')             
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, default='ConvolutionalNet',
                        help='ConvolutionalNet|MLPNet')
parser.add_argument('--dataset', type=str, default='cifar10',
                        help='mnist|cifar10|cifar100|lsun|svhn')
parser.add_argument('--data_path', type=str, default='./data',
                        help='where to save data (if any).')
parser.add_argument('--image_size', type=int, default=32,
                        help='preprocesses data into the specified image size')
parser.add_argument('--save_checkpoint', type=str, default='./checkpoint',
                        help='where to save checkpoints (if any).')
parser.add_argument('--load_checkpoint', type=str,
                        help='where to load checkpoint (if any).')                    
parser.add_argument('--lambd', type=float, default=0.01,
                    help='Regularization Parameter (default: 0.01)')       
parser.add_argument('--reg_type', type=str, default='frobenius_norm', 
                    help='type of regularization: frobenius_norm|pairwise_kl|pairwise_l1|pairwise_l2')
parser.add_argument('--show_activations', type=str, default='OFF',
                        help='Shows layer activations as images ON|OFF')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#####################################################################################
# Load dataset
ALLOWABLE_DATASETS = ['mnist', 'cifar10', 'cifar100', 'lsun', 'svhn']
ALLOWABLE_MODELS = ['ConvolutionalNet', 'MLPNet']

assert args.dataset in ALLOWABLE_DATASETS
assert args.model in ALLOWABLE_MODELS

preprocessing = [
    transforms.Scale(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
]

if args.dataset == 'mnist':
    preprocessing.append(transforms.Normalize((0.1307,), (0.3081,))) 
    train_loader = DataLoader(
        datasets.MNIST(args.data_path, train=True, download=True,
                       transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    test_loader = DataLoader(
        datasets.MNIST(args.data_path, train=False,
                       transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    input_channels = 1
    n_class = 10
elif args.dataset == 'cifar10':
    preprocessing.append(transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5)))
    train_loader = DataLoader(
        datasets.CIFAR10(args.data_path, train=True, download=True,
                         transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    test_loader = DataLoader(
        datasets.CIFAR10(args.data_path, train=False,
                         transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    input_channels = 3
    n_class = 10
elif args.dataset == 'cifar100':
    preprocessing.append(transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2023, 0.1994, 0.2010)))
    train_loader = DataLoader(
        datasets.CIFAR100(args.data_path, train=True, download=True,
                          transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    test_loader = DataLoader(
        datasets.CIFAR100(args.data_path, train=False,
                          transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    input_channels = 3
    n_class = 100
elif args.dataset == 'lsun': # has to be downloaded already
    preprocessing.append(transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5)))
    train_loader = DataLoader(
        datasets.LSUN(args.data_path, 'train',
                      [transforms.Compose(preprocessing), None]),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    test_loader = DataLoader(
        datasets.LSUN(args.data_path, 'test',
                      [transforms.Compose(preprocessing), None]),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    input_channels = 3
    n_class = 10
elif args.dataset == 'svhn':
    preprocessing.append(transforms.Normalize((0.4377, 0.4438, 0.4728),
                                              (0.1980, 0.2010, 0.1970)))
    train_loader = DataLoader(
        datasets.SVHN(args.data_path, split='train', download=True,
                      transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    test_loader = DataLoader(
        datasets.SVHN(args.data_path, split='test',
                      transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=args.cuda)
    input_channels = 3
    n_class = 10

assert train_loader
assert test_loader
assert input_channels
assert n_class

#################################################################################
# Models
class ConvolutionalNet(nn.Module):
    def __init__(self, width, height, depth, n_class):
        super(ConvolutionalNet, self).__init__()
        self.conv1 = nn.Conv2d(depth, 20, 5)
        self.bc1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 20, 3)
        self.bc2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 20, 3)
        self.bc3 = nn.BatchNorm2d(20)
        output_width = width - 8
        output_features = 20 * output_width * output_width
        self.fc1 = nn.Linear(output_features, 50)
        self.bc4 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, n_class)

    def forward(self, x):
        activation_arr = []
        out1 = self.bc1(F.relu(self.conv1(x)))
        activation_arr.append(out1)
        out2 = self.bc2(F.relu(self.conv2(out1)))
        activation_arr.append(out2)
        out3 = self.bc3(F.relu(self.conv3(out2)))
        activation_arr.append(out3)
        x = out3.view(out3.size(0), -1)
        x = self.bc4(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x), activation_arr

    def getRegularizedLoss(self, regularizationMethod='frobenius_norm'):         
        
        result_loss = 0        

        def frobenius_norm(conv): 
            """
            gives frobeniusNorm for a conv layer 
            input: conv layer
            output: loss (scalar)
            """          
            w = conv.weight.view(conv.out_channels, -1)                    
            wwt = w.mm(w.t())
            if args.cuda: 
                I  = Variable(torch.eye(conv.out_channels, conv.out_channels)).cuda()
            else: 
                I  = Variable(torch.eye(conv.out_channels, conv.out_channels))
            loss = torch.norm(wwt-I, 2)
            return loss 

        def pairwise_kl(conv): 
            loss = 0
            w = conv.weight.view(conv.out_channels, -1)
            for i in range(w.size(0)):
                for j in range(w.size(0)):
                    loss += F.kl_div(torch.log(w[i]), w[j], size_average = False)
            return loss

        def pairwise_l1(conv): 
            loss = 0
            w = conv.weight.view(conv.out_channels, -1)
            for i in range(w.size(0)):
                for j in range(w.size(0)):
                    loss += F.l1_loss(w[i], w[j])
            return loss

        def pairwise_l2(conv): 
            loss = 0
            w = conv.weight.view(conv.out_channels, -1)
            for i in range(w.size(0)):
                for j in range(w.size(0)):
                    loss += F.mse_loss(w[i], w[j])
            return loss
            
        if regularizationMethod == 'frobenius_norm':                          
            result_loss += frobenius_norm(self.conv1)
            result_loss += frobenius_norm(self.conv2)
            result_loss += frobenius_norm(self.conv3)            

        elif regularizationMethod == 'pairwise_kl': 
            result_loss += pairwise_kl(self.conv1)
            result_loss += pairwise_kl(self.conv2)
            result_loss += pairwise_kl(self.conv3)

        elif regularizationMethod == 'pairwise_l1': 
            result_loss += pairwise_l1(self.conv1)
            result_loss += pairwise_l1(self.conv2)
            result_loss += pairwise_l1(self.conv3)

        elif regularizationMethod == 'pairwise_l2': 
            result_loss += pairwise_l2(self.conv1)
            result_loss += pairwise_l2(self.conv2)
            result_loss += pairwise_l2(self.conv3)
        
        return result_loss 

class MLPNet(nn.Module):
    def __init__(self, width, height, depth, n_class):
        super(MLPNet, self).__init__()
        self.n_input = width*height*depth
        self.fc1 = nn.Linear(self.n_input, 500)
        self.bc1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 250)
        self.bc2 = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250, n_class)

    def forward(self, x):
        x = x.view((-1, self.n_input))
        x = F.relu(self.bc1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bc2(self.fc2(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

if args.model == "ConvolutionalNet":
    model = ConvolutionalNet(args.image_size, args.image_size, input_channels, n_class)
elif args.model == "MLPNet":
    model = MLPNet(args.image_size, args.image_size, input_channels, n_class)
assert model

if args.cuda:
    model.cuda()

if args.optimizer == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
elif args.optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
assert optimizer

####################################################################################
# Training and Testing
def train(epoch, lambd, reg_type):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        batch_data = data
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output,activations = model(data)      
        if (batch_idx % args.log_interval == 0) and (args.show_activations == "ON"):
            for layer in range(len(activations)):
                label = 'epoch_'+str(epoch)+'_batch_'+str(batch_idx)+'_layer_'+str(layer)
                plot_activations(activations[layer].data.numpy(),label, batch_data)

        loss = F.nll_loss(output, target) + lambd * model.getRegularizedLoss(reg_type)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

            
            if(args.show_activations == 'ON'):
                w = []
            
                w.append(model.conv1.weight.data.numpy())
                w.append(model.conv2.weight.data.numpy())
                w.append(model.conv3.weight.data.numpy())

                for i in range(len(w)):
                    label = 'KernelImg/epoch_'+str(epoch)+'_iter_'+str(batch_idx)+'_layer'+str(i+1)+'.png'
                    plot_kernels(w[i],label)
            

    checkpoint_path = os.path.join(args.save_checkpoint, "%s_epoch_%d.pth" % (args.model, epoch))
    torch.save(model.state_dict(), checkpoint_path)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output,activations = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Creates checkpoint directory if it doesn't exist
if not os.path.exists(args.save_checkpoint):
    os.makedirs(args.save_checkpoint)

start_epoch = 1

# Loads checkpoint if specified
if args.load_checkpoint:
    start_epoch += int(args.load_checkpoint.split('_')[-1][:-4])
    model.load_state_dict(torch.load(args.load_checkpoint))

for epoch in range(start_epoch, args.epochs + 1):
    train(epoch, args.lambd, args.reg_type)
    test()

