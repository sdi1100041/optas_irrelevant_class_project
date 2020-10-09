import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import wandb
import numpy as np
from utils.data_processing.get_dataset import get_train_data, get_validation_data
from utils.models.NN_models import *
from utils.models.CIFAR10_models import *
from utils.utilities.utils import progress_bar
best_acc=0
start_epoch=0
previous_acc=0
ground_truth=0
previous_test_targets_pred=0

def hamming_distance(x,y):
    return np.sum(x!=y)

def define_model(args:dict):
    torch.manual_seed(1274)
    if args['task'] == "MNIST":
        model = CNN_MNIST().cuda()
        args['model'] = 'CNN_MNIST'
    elif args['task'] == "EMNIST":
        model = CNN_EMNIST().cuda()
        args['model'] = 'CNN_EMNIST'
    else: #task == CIFAR10
        if args['model'] == 'DPN92':
            model = DPN92().cuda()
        elif args['model'] == 'ResNet18':
            model = ResNet18().cuda()
        elif args['model'] == 'GoogLeNet':
            model = GoogLeNet().cuda()
        else:
            model = VGG('VGG19').cuda()
            args['model'] = 'VGG19'
    return model

def train(epoch,net,trainloader,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        avg_loss =train_loss/(batch_idx+1)
        avg_accuracy=100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (avg_loss, avg_accuracy, correct, total))

    #wandb.log({"train_loss":avg_loss,"train_accuracy":avg_accuracy},step=epoch)

def test(epoch,net,testloader,criterion,args):
    global best_acc,previous_acc,ground_truth,previous_test_targets_pred
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_targets_pred=np.zeros(testloader.dataset.data.shape[0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            avg_loss =test_loss/(batch_idx+1)
            acc=100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (avg_loss, acc, correct, total))

            test_targets_pred[batch_idx*100:(batch_idx+1)*100] = predicted.cpu().numpy()


    acc=acc/100
    wandb.log({"classification_loss":1-acc},step=epoch)

    if epoch >0:
        wandb.log({"hamming_distance": hamming_distance(previous_test_targets_pred, test_targets_pred)/test_targets_pred.shape[0],"difference":acc - previous_acc },step=epoch)
        print("Test error is", hamming_distance(ground_truth, test_targets_pred)/test_targets_pred.shape[0], 1-acc )
        print("Hamming distance between rounds is",hamming_distance(previous_test_targets_pred, test_targets_pred)/test_targets_pred.shape[0])
        print("Test error of previous epoch is", hamming_distance(ground_truth, previous_test_targets_pred)/ground_truth.shape[0])

    previous_acc=acc
    previous_test_targets_pred=test_targets_pred

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args['task'] + '_'+ args['model']  +'_ckpt.pth')
        best_acc = acc

def construct_and_train(args: dict):
    global start_epoch,best_acc,ground_truth
    trainset = get_train_data(args['task'],args['irr_class'])
    testset = get_validation_data(args['task'],args['irr_class'])
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=128, shuffle= False,num_workers=2)
    testloader = torch.utils.data.DataLoader(testset,batch_size=100, shuffle= False,num_workers=1)
    ground_truth=np.array(testset.targets)

    net=define_model(args)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    wandb.init(project='optas_irrelevant_class_project', name='del_' + str(args['irr_class'])  + '_' + args['task']+'_'+args['model']+'_'+args['algorithm']+'_'+str(args['lr']), config=args)
    wandb.watch(net)

    if args['resume']:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        fl='./checkpoint/' + args['task'] + '_' + args['model'] +'_ckpt.pth'
        assert os.path.exists(fl), 'Error: no checkpoint directory/file found!'
        checkpoint = torch.load(fl)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    if args['algorithm'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    elif args['algorithm'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch + args['epochs']):
        train(epoch,net,trainloader,criterion,optimizer)
        test(epoch,net,testloader,criterion,args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch experiments")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--task", default="MNIST", type=str, choices={"MNIST","EMNIST","CIFAR10"}, help="Dataset")
    parser.add_argument("--model", type=str, choices={'VGG19','ResNet18','GoogLeNet','DPN92'}, help="NN_model")
    parser.add_argument("--epochs",default=20,type=int, choices=range(0,201), help="number of epochs")
    parser.add_argument('--algorithm', type=str, default="SGD", choices={"SGD","Adam"}, help="Optimization algorithm")
    parser.add_argument("--irr_class",default=9,type=int, choices=range(0,10), help="irrelevant class")
    args = vars(parser.parse_args())

    construct_and_train(args)

