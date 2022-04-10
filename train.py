import datetime
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import triple_transforms

from nets import lines_predciton, basic, basic_NL, SANLNet
from config import train_S2B_path, test_S2B_path
from dataset3 import ImageFolder, make_dataset
from misc import AvgMeter, check_mkdir

from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

cudnn.benchmark = True

def main(ckpt_path,
         exp_name,
         iter_num,
         train_batch_size,
         last_iter,
         lr,
         lr_decay,
         weight_decay,
         momentum,
         resume_snapshot,
         val_freq,
         img_size_h,
         img_size_w,
         crop_size,
         snapshot_epochs,
         time):

    ################################ log ################################
    transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    to_pil = transforms.ToPILImage()

    triple_transform = triple_transforms.Compose([
        triple_transforms.Resize((img_size_h, img_size_w)),
        #triple_transforms.RandomCrop(args['crop_size']),
        triple_transforms.RandomHorizontallyFlip()
    ])

    train_set = ImageFolder(train_S2B_path, transform=transform, target_transform=transform, triple_transform=triple_transform, is_train=True)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, num_workers=0, shuffle=True)
    test_list = make_dataset(test_S2B_path, is_train=False)

    criterion = nn.L1Loss()
    log_path = os.path.join(ckpt_path, exp_name, time + '.txt')
    
    net = SANLNet().cuda().train()

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': lr, 'weight_decay': weight_decay}
    ])

    if len(resume_snapshot) > 0:
        print('training resumes from \'%s\'' % resume_snapshot)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, resume_snapshot + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, resume_snapshot + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * lr
        optimizer.param_groups[1]['lr'] = lr
    
    
    ################################ train ################################
    curr_iter = last_iter
    while True:
        train_loss_record = AvgMeter()
        train_structure_loss_record = AvgMeter()
        train_semantic_loss_record = AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * lr * (1 - float(curr_iter) / iter_num
                                                                ) ** lr_decay
            optimizer.param_groups[1]['lr'] = lr * (1 - float(curr_iter) / iter_num
                                                            ) ** lr_decay

            inputs, gts, hough_space_label, gt_coords = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            hough_space_label = Variable(hough_space_label).cuda()

            optimizer.zero_grad()

            result, keypoint_map = net(inputs, train=True)
#            result = net(inputs)
#            print(keypoint_map.size())
#            print(hough_space_label.size())

            loss_structure = criterion(result, gts)
            loss_semantic = torch.nn.functional.binary_cross_entropy_with_logits(keypoint_map, hough_space_label) #dht

            loss = loss_structure + loss_semantic

            loss.backward()

            optimizer.step()

            train_loss_record.update(loss.data, batch_size)
            train_structure_loss_record.update(loss_structure.data, batch_size)
            train_semantic_loss_record.update(loss_semantic.data, batch_size)

            curr_iter += 1

            log = '[iter %d], [total loss %.5f], [lr %.13f], [loss_structure %.5f], [loss_net %.5f]' % \
                  (curr_iter, train_loss_record.avg, optimizer.param_groups[1]['lr'],
                   train_structure_loss_record.avg, train_semantic_loss_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % val_freq == 0:
                MSE, PSNR, SSIM = validate(net, test_list)
                val_log = 'validate: [MSE %.1f], [PSNR %.3f], [SSIM %.5f]' % (MSE, PSNR, SSIM)
                open(log_path, 'a').write('\n' + val_log + '\n\n')

            if (curr_iter + 1) % snapshot_epochs == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, ('%d.pth' % (curr_iter + 1) )))
                torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, ('%d_optim.pth' % (curr_iter + 1) )))

            if curr_iter > iter_num:
                return


def validate(net, test_list):
    print('validating...')
    net.eval()

    transform_test = transforms.Compose([
    transforms.Resize([272,480]),
    transforms.ToTensor() ])
    to_pil = transforms.ToPILImage()

    with torch.no_grad():
        MSE, PSNR, SSIM = 0, 0, 0 

        for idx, path in enumerate(tqdm(test_list)):            
            img_path, gt_path, data_path = path
            data = np.load(data_path, allow_pickle=True).item()
            gt_coords = data["coords"]
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            img_var = Variable(transform_test(img).unsqueeze(0)).cuda()
            
            res = net(img_var, train=False)
            torch.cuda.synchronize()

            ####################### Dehaze ###################### 
            result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))

            gt = Image.open(gt_path).convert('RGB')
            result1 = np.array(result)
            gt = np.array(gt)
            MSE += mean_squared_error(gt, result1)
            PSNR += peak_signal_noise_ratio(gt, result1)
            SSIM += structural_similarity(gt, result1, multichannel=True)

            if idx > 100:
                break

        print("MSE: " + str(MSE / len(img_list)))
        print("PSNR: " + str(PSNR / len(img_list)))
        print("SSIM: " + str(SSIM / len(img_list)))
        return MSE, PSNR, SSIM

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./ckpt', help='model path')
    parser.add_argument('--exp_name', type=str, default='SANLNet', help='model name')
    parser.add_argument('--iter_num', type=int, default=40000, help='iter_num')
    parser.add_argument('--train_batch_size', type=int, default=400, help='train_batch_size')
    parser.add_argument('--last_iter', type=int, default=0, help='last_iter')
    parser.add_argument('--lr', type=int, default=5e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=0.9, help='lr_decay')
    parser.add_argument('--weight_decay', type=int, default=0, help='weight_decay')
    parser.add_argument('--momentum', type=int, default=0.9, help='momentum')
    parser.add_argument('--resume_snapshot', type=str, default='', help='resume_snapshot')
    parser.add_argument('--val_freq', type=int, default=10, help='')
    parser.add_argument('--img_size_h', type=int, default=272, help='')
    parser.add_argument('--img_size_w', type=int, default=480, help='')
    parser.add_argument('--crop_size', type=int, default=512, help='')
    parser.add_argument('--snapshot_epochs', type=int, default=10000, help='')
    parser.add_argument('--time', type=str, default=str(datetime.datetime.now()), help='')
    opt = parser.parse_args()

    check_mkdir(opt.ckpt_path)
    check_mkdir(os.path.join(opt.ckpt_path, opt.exp_name))
    log_path = os.path.join(opt.ckpt_path, opt.exp_name, opt.time + '.txt')
    open(log_path, 'w').write(''.join(f'{k}={v}\n' for k, v in vars(opt).items()) + "\n\n")
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))

