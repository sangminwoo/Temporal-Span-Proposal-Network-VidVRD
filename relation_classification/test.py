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
import json
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num-workers", default=1, type=int)
parser.add_argument("--lr", default=0.01, type=int)
parser.add_argument("--momentum", default=0.9, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=int)
parser.add_argument("--num-classes", default=0.01, type=int)
parser.add_argument("--dump_dir", default="./logs", type=str)
parser.add_argument("--print-freq", default=1, type=int)
parser.add_argument("--save-freq", default=1, type=int)
parser.add_argument("--model_path",
        default="./logs/relation_cls/snapshot/relation_20.pth", type=str)

args = parser.parse_args()
def main():
    model = RelationPredictor()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    test_set = get_testing_set()
    
    test_loader = data.DataLoader(dataset=test_set,
            num_workers=args.num_workers, batch_size=args.batch_size,
            shuffle=False, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)

    cudnn.benchmark = True

    test(test_loader, model, criterion, optimizer)


def test(test_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    log = ""
    sig = torch.nn.Sigmoid()
    final_res_dict = dict()
    for i, pack in enumerate(test_loader):
        sub_features, obj_features, i3d_sub_features, i3d_obj_features, motions, target, meta_info = pack
        vid_name = meta_info[-1][0]
        final_res_dict[vid_name] = []
        print(i, vid_name)
        if len(list(sub_features.shape)) == 2:
            continue
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        sub_var = torch.autograd.Variable(sub_features).cuda()
        obj_var = torch.autograd.Variable(obj_features).cuda()
        i3d_sub_var = torch.autograd.Variable(i3d_sub_features).cuda()
        i3d_obj_var = torch.autograd.Variable(i3d_obj_features).cuda()
        motions = torch.autograd.Variable(motions).cuda()
        target_var = torch.autograd.Variable(target)
        
        feed_list = [sub_var, obj_var, i3d_sub_var, i3d_obj_var, motions]
        with torch.no_grad():
            output = model(*feed_list)
        out = sig(output)
        for i in range(output.shape[1]):
            out[0][i][0] = 0
        k = max(133//output.shape[1], 20)
        res = torch.topk(out, k=k, dim=2)

        for indx in range(output.shape[1]):
            tmp_dict = dict()
            tmp_dict["sub_traj"] = meta_info[0][indx]
            tmp_dict["obj_traj"] = meta_info[1][indx]
            sub_cls = meta_info[2][indx]
            obj_cls = meta_info[3][indx]
            tmp_dict["score"] = meta_info[4][indx].numpy()[0]
            begin = int(meta_info[5][indx][0].numpy()[0])
            end = int(meta_info[5][indx][1].numpy()[0])
            tmp_dict["duration"] = [begin, end]
            rel_cls = []
            rel_scores = res[0][0][indx].cpu().numpy().tolist()
            rel_scores = [float(x) for x in rel_scores]
            rel_cls = res[1][0][indx].cpu().numpy().tolist()
            rel_cls = [int(x) for x in rel_cls]
            tmp_dict["triplet"] = [sub_cls, rel_cls, obj_cls]
            tmp_dict["rel_scores"] = rel_scores
            final_res_dict[vid_name].append(tmp_dict)
    json.dump(final_res_dict, open(os.path.join(args.dump_dir,
        "result.json"), "w"))

def adjust_learning_rate(optimizer, epoch):
    interval = int(args.epochs * 0.4)
    lr = args.lr * (0.1 ** (epoch // 40))
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
