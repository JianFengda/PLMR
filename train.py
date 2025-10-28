import argparse
import os
import shutil
import time
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import tensorboard_logger as tb_logger
from model import *
from utils.utils_loss import *
from utils.cifar10 import load_cifar10
from utils.mnist import load_mnist
from utils.fashionmnist import load_fashionmnist
from utils.kminst import load_kmnist

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str,
                    choices=['cifar10','mnist','fashion','kmnist'],
                    help='dataset name (cifar10)')
parser.add_argument('--exp_dir', default='experiment', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, 
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='600,700,800',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.01,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--loss_weight', default=0.2, type=float,
                    help='Weight of mutual information loss among samples')
parser.add_argument('--loss_weight_c', default=0.1, type=float,
                    help='Cluster center mutual information loss weight')
parser.add_argument('--partial_rate', default=0.1, type=float, 
                    help='ambiguity level (q)')
parser.add_argument('--neigbor_num', default=12, type=int,
                    help='Number of nearest neighbor calculations')
parser.add_argument('--same_weight', default=5, type=float,
                    help='contrastive loss weight')

def main():
    args = parser.parse_args()
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    model_path = 'ds_{ds}_pr_{pr}_lr_{lr}_ep_{ep}_lw_{lw}_lwe_{lwe}_lwec_{lwec}'.format(
                                            ds=args.dataset,
                                            pr=args.partial_rate,
                                            lr=args.lr,
                                            ep=args.epochs,
                                            lw=args.loss_weight,
                                            lwe=args.loss_weight,
                                            lwec=args.loss_weight_c)

    args.exp_dir = os.path.join(args.exp_dir, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    cudnn.benchmark = True
    args.gpu = gpu
    # create model
    print("=> creating model '{}'")
    if args.dataset == 'cifar10':
        train_loader, train_givenY, test_loader = load_cifar10(partial_rate=args.partial_rate, batch_size=args.batch_size,test=False)
    elif args.dataset == 'mnist':
        train_loader, train_givenY, test_loader = load_mnist(partial_rate=args.partial_rate,batch_size=args.batch_size)
    elif args.dataset == 'fashion':
        train_loader, train_givenY, test_loader = load_fashionmnist(partial_rate=args.partial_rate,batch_size=args.batch_size)
    elif args.dataset == 'kmnist':
        train_loader, train_givenY, test_loader = load_kmnist(partial_rate=args.partial_rate,batch_size=args.batch_size)
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

    model = PLMR(num_class=args.num_class, feat_dim=args.low_dim, dataset=args.dataset)
    model = model.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('Calculating uniform targets...')
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float()/tempY
    confidence = confidence.cuda()

    loss_fn = partial_loss()
    loss_cont_fn = MILoss()
    loss_cont_fn_c = CenterMILoss()

    logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir,'tensorboard'), flush_secs=2)

    print('\nStart Training\n')

    best_acc = 0
    tol_time = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        start_time = time.time()
        train(train_loader, confidence,model, loss_fn, loss_cont_fn,loss_cont_fn_c,optimizer, epoch, args, logger)
        end_time = time.time()
        total_time = end_time - start_time
        tol_time = tol_time + total_time

        acc_test = test(model, test_loader, args, epoch, logger)

        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: Acc {}, Best Acc {}. (lr {})\n'.format(epoch
                , acc_test, best_acc, optimizer.param_groups[0]['lr']))
        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
        best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))

def train(train_loader, confidence,model, loss_fn, loss_cont_fn,loss_cont_fn_c,optimizer, epoch, args, tb_logger):
    #定义可以更新的变量
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Center', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_MI_log = AverageMeter('Loss@MI', ':2.2f')
    loss_MI_c_log = AverageMeter('Loss@MI_c', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto, loss_cls_log, loss_MI_log,loss_MI_c_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images_w, images_s, labels, true_labels, index,images_ori) in enumerate(train_loader):
        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()

        cls_out, q, k, am_center_n = model(X_w,X_s)
        features_cont = torch.cat((q, k), dim=0)
        dis_label = disCosine(q, am_center_n)
        logits_prot = torch.mm(q, am_center_n.t())
        score_prot = torch.softmax(logits_prot, dim=1)
        #
        batch_size = cls_out.shape[0]

        images_ori_reshape = images_ori.view(images_ori.size(0), -1).float()
        images_ori_norm = F.normalize(images_ori_reshape, p=2, dim=1)

        sim_matrix = torch.mm(images_ori_norm[:batch_size], images_ori_norm.t())
        _, indices = torch.topk(sim_matrix, k=args.neigbor_num + 1, dim=1)
        indices = indices[:, 1:]
        neighbors_num = torch.full((batch_size, batch_size), float(args.neigbor_num),
                               dtype=torch.float32, device='cuda')

        idx = indices.long().cuda()
        ranks = torch.arange(1, args.neigbor_num + 1, dtype=torch.float32,
                         device='cuda').repeat(batch_size, 1)

        neighbors_num.scatter_(1, idx, ranks)
        mutual_neighbors = neighbors_num + neighbors_num.t()
        same_label_counts = torch.mm(Y, Y.t())

        same_label_result = args.same_weight * same_label_counts
        condition1 = (same_label_result >= mutual_neighbors)
        labels_equal = torch.all(torch.eq(Y.unsqueeze(1), Y.unsqueeze(0)), dim=2)
        condition2 = torch.logical_and(same_label_counts == 1, labels_equal)
        condition3 = (same_label_counts == 0)
        combined_condition = torch.logical_or(condition1, condition2)

        mask = combined_condition.float()
        mask[condition3] = 0

        _, max_indices = torch.max(cls_out, dim= 1)
        mask_c = torch.zeros_like(cls_out)
        mask_c.scatter_(1, max_indices.unsqueeze(1), 1)

        # MI loss
        loss_MI = loss_cont_fn(features_cont,mask=mask, batch_size=batch_size)
        loss_MI_c = loss_cont_fn_c(am_center_n,features=features_cont, mask_c=mask_c,batch_size=batch_size)

        # classification loss
        loss_cls,new_label = loss_fn(cls_out, confidence[index,:], Y_true, dis_label)
        loss = 2*loss_cls + args.loss_weight * loss_MI + args.loss_weight_c * loss_MI_c

        loss_cls_log.update(loss_cls.item())
        loss_MI_log.update(loss_MI.item())
        loss_MI_c_log.update(loss_MI_c.item())

        # log accuracy
        acc = accuracy(cls_out, Y_true)[0]
        acc_cls.update(acc[0])
        acc = accuracy(score_prot, Y_true)[0]
        acc_proto.update(acc[0])


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for j, k in enumerate(index):
            confidence[k, :] = new_label[j, :].detach()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('MI Loss', loss_MI.avg, epoch)
        tb_logger.log_value('MI Center Loss', loss_MI_c.avg, epoch)

def to_logits(y):
    y_=torch.zeros_like(y)
    col = torch.argmax(y,axis=1)
    row = [i for i in range(0,len(y))]
    y_[row,col] = 1
    return y_

def test(model, test_loader, args, epoch, tb_logger):
    with torch.no_grad():
        print('==> Evaluation...')

        model.eval()
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        for batch_idx, (images,labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            images = images.float() if images.dtype is not torch.float else images
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)
        with open('progress.txt', 'a') as file:
            output_string = 'Accuracy is %.2f%% (%.2f%%)' % (acc_tensors[0], acc_tensors[1])
            file.write(output_string + '\n')

        print('Accuracy is %.2f%% (%.2f%%)'%(acc_tensors[0],acc_tensors[1]))
        if args.gpu ==0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('Top5 Acc', acc_tensors[1], epoch)
    return acc_tensors[0]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    main()
