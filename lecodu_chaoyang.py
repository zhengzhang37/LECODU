from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.chaoyang_mr import *
import argparse, sys
import torchvision.models as models
from datetime import datetime
from einops import rearrange
from torch import nn
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def set_device(args):
    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.num_epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_model(args):
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(in_features=512, out_features=args.num_classes)
    # model = torch.load(args.pretrained_path)
    model.load_state_dict(torch.load('../pretrained_models/best_ckpt.pth'))
    cls_net = copy.deepcopy(model)
    cls_net.fc = nn.Linear(in_features=512, out_features=2)
    return model.cuda(), cls_net.cuda()

class CollaborationNet(nn.Module):
    def __init__(self, channels=4, hidden_size=512, dim=4):
        super(CollaborationNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels * dim, hidden_size)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def value_to_one_hot(labels):
    labels = torch.zeros(labels.shape[0], 4).scatter_(1, labels.view(-1, 1), 1)
    return labels

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def generate_experts(labels):
    lst = [0, 1, 2]
    sample_lst = random.sample(lst, 3)
    label1 = value_to_one_hot(labels[:,sample_lst[0]])
    label2 = value_to_one_hot(labels[:,sample_lst[1]])
    label3 = value_to_one_hot(labels[:,sample_lst[2]])
    experts = torch.cat((label1.unsqueeze(1), label2.unsqueeze(1), label3.unsqueeze(1)), dim=1)
    return experts

def train(args, epoch, pred_net, cls_net, fc_net, cross_entropy_loss, l1_loss, optimizer, train_loader):
    cls_net.train()
    fc_net.train()
    num_iter = (len(train_loader.dataset) // args.batch_size) + 1

    # # load the pretrained network
    with torch.no_grad():
        pred_net = pred_net.cuda()

    for batch_idx, (inputs, labels, index) in enumerate(train_loader):
        
        # input, input w. random aug, label
        inputs = inputs.cuda()
        labels = labels
        with torch.no_grad():
            # calculate the pretrained models' prediction
            out_ai = pred_net(inputs)

        # define utility matrix cost & hyper parameter
        utility_matrix = torch.tensor([[0.0, 1.0]]).cuda()
        utility_matrix *= args.t

        # expert 1,2,3 and pseudo label 1,2
        pseudo_label = labels[:, 1].cuda()
        experts = generate_experts(labels).cuda() # [256, 10]

        # HCI network and gumbel softmax -> users selection and model prediction
        out_prob = cls_net(inputs)
        deferal = gumbel_softmax_sample(out_prob, args.temp1)
        pred_ai = gumbel_softmax_sample(out_ai, args.temp2)

        # combine AI and human predictions
        decision = torch.cat((pred_ai.unsqueeze(1), experts), dim=1)

        # # make decision
        defer = deferal @ torch.tril(torch.ones(2, 2),diagonal=0).cuda()
        decision = defer.unsqueeze(-1) * decision
        output = fc_net(decision)

        # # calculate utility cost
        utility_weight = utility_matrix @ torch.t(deferal)

        output = fc_net(decision)
        # calculate loss function
        ce_loss = cross_entropy_loss(output, pseudo_label)

        if args.defer == True: 
            loss = ce_loss + utility_weight.mean()
        else:
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0 :
            print('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Net loss: %.2f'
                        % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))

def test(epoch, pred_net, cls_net, fc_net, test_loader, test_log):
    cls_net.eval()
    fc_net.eval()
    pred_net.eval()
    correct = 0
    pseudo_correct = 0
    total = 0
    counts = np.zeros((4)).astype(float)

    with torch.no_grad():

        # load the pretrained network
        pred_net = pred_net.cuda()

        for batch_idx, (inputs, targets, users) in enumerate(test_loader):

            # input, true label
            inputs, targets = inputs.cuda(), targets.cuda()

            # calculate the pretrained models' prediction
            out_ai = pred_net(inputs)

            # expert 1,2,3
            # print(users.shape)
            experts = generate_experts(users).cuda() # [256, 10]

            # HCI network and gumbel softmax -> users selection and model prediction
            out_prob = cls_net(inputs)
            deferal = gumbel_softmax_sample(out_prob, args.temp1)
            pred_ai = gumbel_softmax_sample(out_ai, args.temp2)

            # combine AI and human predictions
            decision = torch.cat((pred_ai.unsqueeze(1), experts), dim=1)
            
            # make decision
            defer = deferal @ torch.tril(torch.ones(2, 2),diagonal=0).cuda()
            decision = defer.unsqueeze(-1) * decision
            output = fc_net(decision)

            _, predicted = torch.max(output, 1)


            ori_pred_net, _ = create_model(args)
            ori_pred_net = ori_pred_net.cuda()
            ori_pred_net.eval()
            pseudo_label = ori_pred_net(inputs)
            _, pseudo_predicted = torch.max(pseudo_label, 1)


            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            pseudo_correct += pseudo_predicted.eq(targets).cpu().sum().item()
            choice_index = deferal.detach().cpu().numpy()

            for i in range(len(choice_index)):
                for j in range(len(choice_index[0])):
                    counts[j] = counts[j] + choice_index[i][j]
    cost = 0
    for i in range(len(counts)):
        cost += i * counts[i]
    cost = cost / total * 10000
    acc = 100. * correct / total
    acc_ori = 100. * pseudo_correct / total
    print("| Test \t Acc Net: %.2f%%, Ori Acc Net: %.2f%%, Cost: %d" % (acc, acc_ori, cost))
    print("user choice network:%d,%d,%d,%d \n\n" % (counts[0], counts[1],counts[2],counts[3]))
    test_log.write('Epoch:%d   Accuracy:%.2f, Cost:%d\n' % (epoch, acc, cost))
    test_log.write("user choice network:%d,%d,%d,%d \n\n" % (counts[0], counts[1],counts[2],counts[3]))
    test_log.flush()

    return acc

def main(args):
    # set_device(args)
    time = datetime.now()
    date_time_str = time.strftime("%Y%m%d_%H%M%S")
    version = args.version
    save_path = os.path.join(args.model_save_path, args.version)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # Logs
    stats_log = os.path.join(save_path, 'stats.txt')
    test_log = open(os.path.join(save_path, 'acc.txt'), 'w')
    with open(stats_log, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n'%(key, value))
    print('| Building net')
    pred_net, cls_net = create_model(args)
    if args.fc_model == 'fc':
        fc_net = CollaborationNet().cuda()
    cudnn.benchmark = True

    optimizer = optim.SGD([{'params': cls_net.parameters()},
                           {'params': fc_net.parameters()}
                        ], lr=args.lr, momentum=0.9, weight_decay=5e-4)

    
    transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(), 
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()])
    transform_test = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        ])
    train_dataset = CHAOYANG(transform=transform_train, mode="human",)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)
    test_dataset = CHAOYANG(transform=transform_test, mode='human_test')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)

    cross_entropy_loss = nn.CrossEntropyLoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    best_acc = 0

    for epoch in range(args.num_epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, epoch, pred_net, cls_net, fc_net, cross_entropy_loss, l2_loss, optimizer, train_loader)
        cur_acc = test(epoch, pred_net, cls_net, fc_net, test_loader, test_log)
        if best_acc < cur_acc:
            torch.save(cls_net, os.path.join(save_path, "gating_best.pth.tar"))
            torch.save(fc_net, os.path.join(save_path, "collaboration_best.pth.tar"))
            best_acc = cur_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Chaoyang Training')
    parser.add_argument('--dataset', type = str, default = 'chaoyang')
    parser.add_argument('--batch_size', default=96, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--cosine', action='store_true', default=False, help='use cosine lr schedule')
    parser.add_argument('--t', default=0.04, type=float, help='utility_matrix hyterparameter')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--temp1', type=float, default=5, metavar='S', help='tau(temperature) (default: 1.0)')
    parser.add_argument('--temp2', type=float, default=0.5, metavar='S', help='tau(temperature) (default: 1.0)')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--warmup_ep', default=50, type=int, help = 'parameter ramp-up epoch')
    parser.add_argument('--pretrained_path', type=str, help='path of pretrained model', default='')
    parser.add_argument('--model_save_path', default='', type=str)
    parser.add_argument('--fc_model', default='fc', type=str)
    parser.add_argument('--version', default='cost_test', type=str)
    parser.add_argument('--defer', default=True, type=bool)
    parser.add_argument('--num_workers', type=int, default=8, help='how many subprocesses to use for data loading')
    args = parser.parse_args()
    print(args)
    set_device(args)
    main(args)

    

    
    

