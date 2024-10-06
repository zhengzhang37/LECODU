from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
import dataloader_cifar as dataloader
from model import *
from utils.utils import *
from datetime import datetime
import torch.distributions as dist
from einops import rearrange
from torch import nn
from cleanlab.multiannotator import get_label_quality_multiannotator
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def set_device(args):
    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def create_model(args):
    model = DualNet(args.num_class)
    model = model.cuda()
    model = torch.load(args.pretrained_path)
    return model

class CollaborationNet(nn.Module):
    def __init__(self, channels=101, hidden_size=512, dim=10):
        super(CollaborationNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels * dim, hidden_size)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class GatingNet(nn.Module):
    def __init__(self, num_class, args):
        super(GatingNet, self).__init__()
        self.pretrained_model = create_model(args).cuda()
        self.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x, index):
        if index == 1:
            self.feature_extractor = self.pretrained_model.net1
        else:
            self.feature_extractor = self.pretrained_model.net2
        x = self.feature_extractor.avgpool(self.feature_extractor.layer4(self.feature_extractor.layer3(self.feature_extractor.layer2(self.feature_extractor.layer1(self.feature_extractor.bn1(self.feature_extractor.conv1(x)))))))
        feature = torch.flatten(x, 1)
        out_ai = self.feature_extractor.pseudo_linear(feature)
        out_prob = self.fc(feature)
        
        return out_ai, out_prob

def value_to_one_hot(labels_x):
    labels = torch.zeros(labels_x.shape[0], 10).scatter_(1, labels_x.view(-1, 1), 1)
    return labels

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def generate_experts(humans, num_classes):
    experts = []
    b, n = humans.size()
    lst = list(range(n))
    random.shuffle(lst)
    sample_lst = lst
    expert_list = []
    for idx in sample_lst:
        expert_list.append(F.one_hot(humans[:, idx], num_classes=num_classes).float())
    experts = torch.cat([expert.unsqueeze(1) for expert in expert_list], dim=1)
    return experts

def train(args, epoch, pred_net, cls_net, fc_net, cross_entropy_loss, optimizer, train_loader):
    cls_net.train()
    fc_net.train()
    num_iter = (len(train_loader.dataset) // args.batch_size) + 1

    # load the pretrained network
    with torch.no_grad():
        pred_net1 = pred_net.net1.cuda()
        pred_net2 = pred_net.net2.cuda()

    for batch_idx, (inputs_x, inputs_x2, labels_x, humans, index) in enumerate(train_loader):
        
        # input, input w. random aug, label
        inputs_x = inputs_x.cuda()
        inputs_x2 = inputs_x2.cuda()
        labels_x = labels_x
        
        # calculate the pretrained models' prediction
        out_ai_1 = pred_net1(inputs_x)
        out_ai_2 = pred_net2(inputs_x2)

        # define utility matrix cost & hyper parameter
        utility_matrix = torch.tensor(list(range(args.num_users + 1)) + list(range(1, args.num_users + 1)), dtype=torch.float32).view(1, -1).cuda()
        utility_matrix *= args.t
        # expert 1,2,3 and pseudo label 1,2
        pseudo_label_1 = labels_x[:, 0].cuda()
        pseudo_label_2 = labels_x[:, 1].cuda()
        experts = generate_experts(humans, 10).cuda() # [256, 10]

        # HCI network and gumbel softmax -> users selection and model prediction
        _, out_prob_1 = cls_net(inputs_x, 1)
        deferal_1 = gumbel_softmax_sample(out_prob_1, args.temp1)
        pred_ai_1 = gumbel_softmax_sample(out_ai_1, args.temp2)

        _, out_prob_2 = cls_net(inputs_x2, 2)
        deferal_2 = gumbel_softmax_sample(out_prob_2, args.temp1)
        pred_ai_2 = gumbel_softmax_sample(out_ai_2, args.temp2)

        # combine AI and human predictions
        decision_1 = torch.cat((pred_ai_1.unsqueeze(1), experts), dim=1)
        decision_2 = torch.cat((pred_ai_2.unsqueeze(1), experts), dim=1)

        weighted_matrix = torch.zeros((args.num_users*2+1, args.num_users+1), dtype=torch.float32)
        weighted_matrix[:args.num_users+1,:] = torch.tril(torch.ones(args.num_users+1, args.num_users+1),diagonal=0)
        for i in range(args.num_users):
            weighted_matrix[i+args.num_users+1, 1:i+2] = 1
        # make decision
        defer_1 = deferal_1 @ weighted_matrix.cuda()
        decision_1 = defer_1.unsqueeze(-1) * decision_1
        
        output_1 = fc_net(decision_1)
        defer_2 = deferal_2 @ weighted_matrix.cuda()
        decision_2 = defer_2.unsqueeze(-1) * decision_2
        output_2 = fc_net(decision_2)

        # calculate utility cost
        utility_weight1 = utility_matrix @ torch.t(deferal_1)
        utility_weight2 = utility_matrix @ torch.t(deferal_2)

        output_1 = fc_net(decision_1)
        output_2 = fc_net(decision_2)
        # calculate loss function
        ce_loss_1 = cross_entropy_loss(output_1, pseudo_label_2)
        ce_loss_2 = cross_entropy_loss(output_2, pseudo_label_1)

        if args.defer == True: 
            loss = ce_loss_1 + ce_loss_2 + utility_weight2.mean() + utility_weight1.mean()
        else:
            loss = ce_loss_1 + ce_loss_2 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0 :
            print('%s:%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Net loss: %.2f'
                        % (args.dataset, args.noise_type, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))

def value_to_one_hot(labels_x):
    labels = torch.zeros(labels_x.shape[0], 10).scatter_(1, labels_x.view(-1, 1), 1)
    return labels

def test(epoch, pred_net, cls_net, fc_net, test_loader, test_log):
    cls_net.eval()
    fc_net.eval()
    correct = 0
    correct2 = 0
    correctmean = 0
    total = 0
    counts1 = np.zeros((args.num_users*2 + 1)).astype(float)
    counts2 = np.zeros((args.num_users*2 + 1)).astype(float)

    with torch.no_grad():

        # load the pretrained network
        pred_net1 = pred_net.net1.cuda()
        pred_net2 = pred_net.net2.cuda()

        for batch_idx, (inputs, targets, experts) in enumerate(test_loader):

            # input, true label
            inputs, targets = inputs.cuda(), targets.cuda()

            # calculate the pretrained models' prediction
            out_ai_1 = pred_net1(inputs)
            out_ai_2 = pred_net2(inputs)

            experts = generate_experts(experts, 10).cuda() # [256, 10]
            # HCI network and gumbel softmax -> users selection and model prediction
            _, out_prob_1 = cls_net(inputs, 1)
            deferal_1 = gumbel_softmax_sample(out_prob_1, args.temp1)
            pred_ai_1 = gumbel_softmax_sample(out_ai_1, args.temp2)

            _, out_prob_2 = cls_net(inputs, 2)
            deferal_2 = gumbel_softmax_sample(out_prob_2, args.temp1)
            pred_ai_2 = gumbel_softmax_sample(out_ai_2, args.temp2)

            decision_1 = torch.cat((pred_ai_1.unsqueeze(1), experts), dim=1)
            decision_2 = torch.cat((pred_ai_2.unsqueeze(1), experts), dim=1)
            
            # make decision
            weighted_matrix = torch.zeros((args.num_users*2+1, args.num_users+1), dtype=torch.float32)
            weighted_matrix[:args.num_users+1,:] = torch.tril(torch.ones(args.num_users+1, args.num_users+1),diagonal=0)
            for i in range(args.num_users):
                weighted_matrix[i+args.num_users+1, 1:i+2] = 1
            # # make decision
            defer_1 = deferal_1 @ weighted_matrix.cuda()
            decision_1 = defer_1.unsqueeze(-1) * decision_1
            output_1 = fc_net(decision_1)
            defer_2 = deferal_2 @ weighted_matrix.cuda()
            decision_2 = defer_2.unsqueeze(-1) * decision_2
            output_2 = fc_net(decision_2)

            _, predicted_1 = torch.max(output_1, 1)
            _, predicted_2 = torch.max(output_2, 1)

            outputs_mean = (output_1 + output_2) / 2
            _, predicted_mean = torch.max(outputs_mean, 1)

            total += targets.size(0)
            correct += predicted_1.eq(targets).cpu().sum().item()
            correct2 += predicted_2.eq(targets).cpu().sum().item()
            correctmean += predicted_mean.eq(targets).cpu().sum().item()

            choice_index_1 = deferal_1.detach().cpu().numpy()
            choice_index_2 = deferal_2.detach().cpu().numpy()

            for i in range(len(choice_index_1)):
                for j in range(len(choice_index_1[0])):
                    counts1[j] = counts1[j] + choice_index_1[i][j]
                    counts2[j] = counts2[j] + choice_index_2[i][j]
    cost = 0
    for i in range(len(counts1)):
        if i <= args.num_users:
            cost += i * (counts1[i] + counts2[i])
        else:
            cost += (i-args.num_users) * (counts1[i] + counts2[i])
    acc = 100. * correct / total
    acc2 = 100. * correct2 / total
    accmean = 100. * correctmean / total
    counts_mean = ((counts1+counts2)/2).astype(int)
    print("| Test \t Acc Net: %.2f%%, Acc Net2: %.2f%%, Acc Mean: %.2f%%, Cost: %d" % (acc, acc2, accmean, cost))
    print("user choice network1:", counts1[:10].astype(int), " user choice network2:", counts2[:10].astype(int))
    test_log.write('Epoch:%d   Accuracy:%.2f, Cost:%d\n' % (epoch, accmean, cost))
    test_log.write("user choice network: %d,%d,%d,%d,%d,%d,%d\n" % (counts_mean[0], counts_mean[1], counts_mean[2], counts_mean[3], counts_mean[4], counts_mean[5], counts_mean[6]))
    test_log.flush()
    return accmean

def main(args):
    save_path = os.path.join(args.model_save_path, args.version)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Logs
    stats_log = os.path.join(save_path, 'stats.txt')
    test_log = open(os.path.join(save_path, 'acc.txt'), 'w')
    with open(stats_log, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n'%(key, value))
    # Hyper Parameters
    noise_type_map = {'human': 'human'}
    args.noise_type = noise_type_map[args.noise_type]

    # load dataset
    # please change it to your own datapath
    if args.data_path is None:
        args.data_path = '../cifar-10h/cifar-10-batches-py'
    
    # please change it to your own datapath for CIFAR-N
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = '../cifar-10h/CIFAR-10_human.pt'
    
    print('| Building net')
    pred_net = create_model(args)
    cls_net = GatingNet(num_class=2 * args.num_users + 1, args=args).cuda()
    fc_net = CollaborationNet(channels=args.num_users + 1).cuda()
    cudnn.benchmark = True

    optimizer = optim.SGD([{'params': cls_net.parameters()},
                           {'params': fc_net.parameters()}
                        ], lr=args.lr, momentum=0.9, weight_decay=5e-4)

    loader = dataloader.cifar_dataloader(dataset='cifar',noise_type = 'idn',batch_size=1024,num_workers=8,user_nums=args.num_users,r = 0.2)
    train_loader = loader.run('train') 
    test_loader = loader.run('test')

    cross_entropy_loss = nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        # start =datetime.datetime.now()
        train(args, epoch, pred_net, cls_net, fc_net, cross_entropy_loss, optimizer, train_loader)
        # end=datetime.datetime.now()
        # print('Running time: %s Seconds'%(end-start).seconds)

        # start =datetime.datetime.now()
        cur_acc = test(epoch, pred_net, cls_net, fc_net, test_loader, test_log)
        # end=datetime.datetime.now()
        # print('Running time: %s Seconds'%(end-start).seconds)

        if best_acc < cur_acc:
            torch.save(cls_net, os.path.join(save_path, "gating_best.pth.tar"))
            torch.save(fc_net, os.path.join(save_path, "collaboration_best.pth.tar"))
            best_acc = cur_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--cosine', action='store_true', default=False, help='use cosine lr schedule')
    parser.add_argument('--noise_type', type=str, default='human')
    parser.add_argument('--noise_path', type=str, help='path of CIFAR-10_human.pt', default=None)
    parser.add_argument('--t', default=0.3, type=float, help='utility_matrix hyterparameter')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--temp1', type=float, default=5, metavar='S', help='tau(temperature) (default: 1.0)')
    parser.add_argument('--temp2', type=float, default=0.5, metavar='S', help='tau(temperature) (default: 1.0)')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--data_path', default=None, type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--is_human', action='store_true', default=False)
    parser.add_argument('--warmup_ep', default=10, type=int, help = 'parameter ramp-up epoch')
    parser.add_argument('--noise_mode', default='cifarn', type=str,help='cifarn, sym, asym')
    parser.add_argument('--pretrained_path', type=str, help='path of pretrained model', default='../cifar10nh_checkpoints/cifar10_random_label1_best.pth.tar')
    parser.add_argument('--model_save_path', default='../idn30', type=str)
    parser.add_argument('--fc_model', default='fc', type=str)
    parser.add_argument('--version', default='moe003', type=str)
    parser.add_argument('--defer', default=True, type=bool)
    parser.add_argument('--num_users', default=3, type=int)
    args = parser.parse_args()
    print(args)
    set_device(args)