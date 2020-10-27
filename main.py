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
irr_class=10
regularization =False
indices=0

def calculate_correct(predicted,targets):
    global irr_class
    rel_class_indices = (targets < irr_class)

    return torch.sum(targets[rel_class_indices] == predicted[rel_class_indices] )

def my_cross_entropy(output,targets):
    global irr_class, regularization
    output = nn.LogSoftmax(dim=1)(output)
    batch_size=targets.shape[0]
    rel_class_indices = (targets < irr_class)
    irr_class_output = output[~rel_class_indices]
    output = output[rel_class_indices]
    targets = targets[rel_class_indices]

    logs=torch.gather(output,1,targets.unsqueeze(1)).squeeze()

    if not regularization:
        return (targets.shape[0],batch_size -targets.shape[0], -torch.mean(logs), torch.tensor(0.0,requires_grad=True))

    irr_class_logs= torch.mean(irr_class_output,dim=1)
    return (targets.shape[0],batch_size -targets.shape[0] , -torch.sum(logs)/float(batch_size), -torch.sum(irr_class_logs)/float(batch_size))

def hamming_distance(x,y):
    return np.sum(x!=y)

def get_gradient( model: torch.nn.Module):
    return torch.cat([x.grad.data.view(-1) for x in model.parameters()])

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
    global irr_class,regularization
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    sum_norm_grad_rel=0.0
    sum_norm_grad_irr=0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        net.zero_grad()
        outputs = net(inputs)
        (num_rel,num_irr,loss_rel,loss_irr) = criterion(outputs, targets)
        num_tot=num_rel+num_irr
        loss_rel.backward(retain_graph=True)
        grad_rel= get_gradient(net)
        loss_irr *= 10
        loss_irr.backward()
        grad_tot= get_gradient(net)
        sum_norm_grad_rel +=torch.norm(grad_rel)*num_tot/float(num_rel)
        sum_norm_grad_irr +=torch.norm(grad_tot - grad_rel)*num_tot/float(num_irr)

        optimizer.step()

        train_loss += (loss_rel + loss_irr).item()
        _, predicted = outputs.max(1)
        total += torch.sum(targets < irr_class).item()
        correct += calculate_correct(predicted,targets)
        avg_loss =train_loss/(batch_idx+1)
        avg_accuracy=100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (avg_loss, avg_accuracy, correct, total))

    wandb.log({"Avg_norm_gradient_relevant":sum_norm_grad_rel/float(len(trainloader)),"Avg_norm_gradient_irrelevant":sum_norm_grad_irr/float(len(trainloader))},step=epoch)

def test(epoch,net,testloader,criterion,args):
    global best_acc,previous_acc,ground_truth,previous_test_targets_pred,irr_class, indices
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_targets_pred=np.zeros(testloader.dataset.data.shape[0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            (num_rel,num_irr,loss_rel,loss_irr) = criterion(outputs, targets)
            loss = (loss_rel  + loss_irr) #/float(num_rel + num_irr)


            test_loss += loss.item()
            outputs = nn.Softmax(dim=1)(outputs)
            max_probs, predicted = outputs.max(1)
            #min_probs, _ = outputs.min(1)
            #print("Maximum and minimum probabilities for irrelevant class",max_probs[targets==9],min_probs[targets==9])
            #for i,d in enumerate( max_probs - min_probs):
            #    if d < 0.2:
            #        predicted[i]=9
            total += torch.sum(targets < irr_class).item()
            correct += calculate_correct(predicted,targets).item()
            avg_loss =test_loss/(batch_idx+1)
            acc=100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (avg_loss, acc, correct, total))

            test_targets_pred[batch_idx*100:(batch_idx+1)*100] = predicted.cpu().numpy()


    acc=acc/100
    wandb.log({"classification_loss":1-acc},step=epoch)
    #test_targets_pred = test_targets_pred[indices]

    if epoch >0:
        #wandb.log({"hamming_distance": hamming_distance(previous_test_targets_pred, test_targets_pred)/test_targets_pred.shape[0],"difference":acc - previous_acc },step=epoch)
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
    global start_epoch,best_acc,ground_truth,indices
    trainset = get_train_data(args['task'])
    testset = get_validation_data(args['task'])
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=128, shuffle= True,num_workers=2)
    testloader = torch.utils.data.DataLoader(testset,batch_size=100, shuffle= False,num_workers=1)
    ground_truth=np.array(testset.targets)

    net=define_model(args)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    prefix = ('REG_10x_') if args['regularization'] else ('NOT_REG_')
    wandb.init(project='1_out_of_5_optas_irrelevant_class_project', name=prefix + args['task']+'_'+args['model']+'_'+args['algorithm']+'_'+str(args['lr']), config=args)
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

    #criterion = nn.CrossEntropyLoss()
    criterion = my_cross_entropy
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
    parser.add_argument("--task", default="MNIST", type=str, choices={"MNIST","EMNIST","CIFAR10","CIFAR100"}, help="Dataset")
    parser.add_argument("--model", type=str, choices={'VGG19','ResNet18','GoogLeNet','DPN92'}, help="NN_model")
    parser.add_argument("--epochs",default=20,type=int, choices=range(0,201), help="number of epochs")
    parser.add_argument("--irr_class",default=10,type=int, choices=range(0,11), help="irrelevant class")
    parser.add_argument( "--regularization", action='store_true', help="use regularization")
    parser.add_argument('--algorithm', type=str, default="SGD", choices={"SGD","Adam"}, help="Optimization algorithm")
    args = vars(parser.parse_args())
    args['irr_class'] = 10
    irr_class = args['irr_class']
    regularization = args['regularization']
    construct_and_train(args)
