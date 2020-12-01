import argparse
import os
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from utils import g, AverageMeter, load_source

from datasets import get_training_set, get_testing_set
from model import RelationPredictor
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num-workers", default=8, type=int)
parser.add_argument("--lr", default=0.1, type=int)
parser.add_argument("--momentum", default=0.9, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=int)
parser.add_argument("--num-classes", default=134, type=int)
parser.add_argument("--dump_dir", default="./logs/relation_cls", type=str)
parser.add_argument("--print-freq", default=50, type=int)
parser.add_argument("--save-freq", default=1, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dump_dir):
    os.makedirs(args.dump_dir)
def main():
    model = RelationPredictor()
    model.cuda()
    train_set = get_training_set()
    
    train_loader = data.DataLoader(dataset=train_set,
            num_workers=args.num_workers, batch_size=args.batch_size,
            shuffle=True, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)

    cudnn.benchmark = True

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        if epoch % args.save_freq == 0:
            save_dir = os.path.join(args.dump_dir, "snapshot")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            filename = os.path.join(args.dump_dir, "snapshot", "relation_%d.pth" %
                epoch)
            is_best = False
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, is_best, filename)
        

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    log = ""
    for i, pack in enumerate(train_loader):
        sub_features, obj_features, i3d_sub_features, i3d_obj_features, motion, target, weight = pack
        if len(list(sub_features.shape)) == 2:
            continue
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        sub_var = torch.autograd.Variable(sub_features).cuda()
        obj_var = torch.autograd.Variable(obj_features).cuda()
        i3d_sub_var = torch.autograd.Variable(i3d_sub_features).cuda()
        i3d_obj_var = torch.autograd.Variable(i3d_obj_features).cuda()
        motion = torch.autograd.Variable(motion).cuda()
        target_var = torch.autograd.Variable(target)
        weight = weight.cuda()
        
        feed_list = [sub_var, obj_var, i3d_sub_var, i3d_obj_var, motion]
        output = model(*feed_list)
        loss = criterion(output, target_var)
        
        losses.update(loss.data, sub_var.size(0)*sub_var.size(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            log = ("Epoch: [{0}][{1}/{2}]\t"
                   "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                   "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                   "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                   epoch, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            with open(os.path.join(args.dump_dir, "log.txt"), "a+") as f:
                f.writelines(log + "\n")
                f.close()
            print(log)
    with open(os.path.join(args.dump_dir, "state.txt"), "a+") as f:
        f.writelines(log + "\n")
        f.close()

def adjust_learning_rate(optimizer, epoch):
    interval = int(args.epochs * 0.4)
    lr = args.lr * (0.1 ** (epoch // interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_checkpoint(state, is_best, filename):
    try:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.dump_dir,
                "relation_best.pth"))
    except Exception as e:
        print("save error")

if __name__ == "__main__":
    main()
