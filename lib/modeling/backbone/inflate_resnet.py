import argparse
import copy
import json

from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from i3res import I3ResNet

# To profile uncomment @profile and run `kernprof -lv inflate_resnet.py`
# @profile
def run_inflater(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder('/home/t2_u1/data/vidvrd/image/',
                                   transforms.Compose([
                                       transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    # class_idx = {0: 'airplane', 1: 'antelope', 2: 'ball', 3: 'bear', 4: 'bicycle', 5: 'bird', 6: 'bus', 7: 'car', 8: 'cattle', 9: 'dog', 10: 'domestic_cat', 11: 'elephant', 12: 'fox', 13: 'frisbee', 14: 'giant_panda', 15: 'hamster', 16: 'horse', 17: 'lion', 18: 'lizard', 19: 'monkey', 20: 'motorcycle', 21: 'person', 22: 'rabbit', 23: 'red_panda', 24: 'sheep', 25: 'skateboard', 26: 'snake', 27: 'sofa', 28: 'squirrel', 29: 'tiger', 30: 'train', 31: 'turtle', 32: 'watercraft', 33: 'whale', 34: 'zebra'}
    # classes = [category for category in class_idx.values()]

    class_idx = json.load(open('imagenet_class_index.json'))
    classes = [class_idx[str(k)][1] for k in range(len(class_idx))]

    if args.resnet_nb == 50:
        resnet = torchvision.models.resnet50(pretrained=True)
    elif args.resnet_nb == 101:
        resnet = torchvision.models.resnet101(pretrained=True)
    elif args.resnet_nb == 152:
        resnet = torchvision.models.resnet152(pretrained=True)
    else:
        raise ValueError('resnet_nb should be in [50|101|152] but got {}'
                         ).format(args.resnet_nb)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
    i3resnet = I3ResNet(copy.deepcopy(resnet), args.frame_nb)
    i3resnet.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3resnet = i3resnet.to(device)
    resnet = resnet.to(device)

    for i, (input_2d, target) in enumerate(loader):
        target = target.to(device)
        target_var = torch.autograd.Variable(target)
        input_2d_var = torch.autograd.Variable(input_2d.to(device))

        out2d = resnet(input_2d_var)
        out2d = out2d.cpu().data

        input_3d = input_2d.unsqueeze(2).repeat(1, 1, args.frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d.to(device))

        out3d = i3resnet(input_3d_var)
        out3d = out3d.cpu().data

        out_diff = out2d - out3d
        print('mean abs error {}'.format(out_diff.abs().mean()))
        print('mean abs val {}'.format(out2d.abs().mean()))

        # Computing errors between final predictions of inflated and uninflated
        # dense networks
        print(
            'Batch {i} maximum error between 2d and inflated predictions: {err}'.
            format(i=i, err=out_diff.max()))
        assert (out_diff.max() < 0.0001)

        if args.display_samples:
            max_vals, max_indexes = out3d.max(1)
            for sample_idx in range(out3d.shape[0]):
                sample_out = out3d[sample_idx]

                top_val, top_idx = torch.sort(sample_out, 0, descending=True)

                print('Top {} classes and associated scores: '.format(
                    args.top_k))
                for i in range(args.top_k):
                    print('[{}]: {}'.format(classes[top_idx[i]],
                                            top_val[i]))

                sample_img = input_2d[sample_idx].numpy().transpose(1, 2, 0)
                sample_img = (sample_img - sample_img.min()) * (1 / (
                    sample_img.max() - sample_img.min()))
                plt.imshow(sample_img)
                plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Inflates ResNet and runs\
    it on dummy dataset to compare outputs from original and inflated networks\
    (should be the same)')
    parser.add_argument(
        '--resnet_nb',
        type=int,
        default=101,
        help='What version of ResNet to use, in [50|101|152]')
    parser.add_argument(
        '--display_samples',
        action='store_true',
        help='Whether to display samples and associated\
        scores for 3d inflated resnet')
    parser.add_argument(
        '--top_k',
        type=int,
        default='5',
        help='When display_samples, number of top classes to display')
    parser.add_argument(
        '--frame_nb',
        type=int,
        default='16',
        help='Number of video_frames to use (should be a multiple of 8)')
    args = parser.parse_args()
    run_inflater(args)