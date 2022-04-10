import os
import time
import argparse

import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from nets import basic, depth_predciton, basic_NL, DGNLNet
from config import test_raincityscapes_path
from misc import check_mkdir
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from dataset3 import ImageFolder, make_dataset
from skimage.measure import label, regionprops
from jittor_code.utils import reverse_mapping, visulize_mapping, edge_align, get_boundary_point
from hungarian_matching import caculate_tp_fp_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
args = {
    'snapshot': '40000',
    'depth_snapshot': ''
}

transform = transforms.Compose([
    transforms.Resize([272,480]),
    transforms.ToTensor() ])


to_pil = transforms.ToPILImage()


def predict(exp_name, root_input, threshold):
    net = DGNLNet().cuda()

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                       map_location=lambda storage, loc: storage.cuda(0)))

    net.eval()
    avg_time = 0

    save_path = os.path.join(ckpt_path, exp_name, '(%s) prediction_%s_%s' % (exp_name, root_input.split("/")[-1], args['snapshot']))
    check_mkdir(save_path)

    with torch.no_grad():
        img_list = [img_name for img_name in os.listdir(root_input)]

        for idx, img_name in enumerate(img_list):           
            img = Image.open(os.path.join(root_input, img_name)).convert('RGB')
            w, h = img.size
            img_var = Variable(transform(img).unsqueeze(0)).cuda()

            start_time = time.time()

            res, key_points = net(img_var)

            torch.cuda.synchronize()
            
            avg_time = avg_time + time.time() - start_time
            print('predicting: %d / %d, avg_time: %.5f' % (idx + 1, len(img_list), avg_time/(idx+1)))
            result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))
            result.save(os.path.join(save_path, img_name))


            ################### DHT ##########################
            key_points = torch.sigmoid(key_points)
            binary_kmap = key_points.squeeze().cpu().numpy() > threshold
            kmap_label = label(binary_kmap, connectivity=1)
            props = regionprops(kmap_label)
            plist = []
            for prop in props:
                plist.append(prop.centroid)

            size = (272, 480)
            b_points = reverse_mapping(plist, numAngle=100, numRho=100, size=(272, 480))
            scale_w = size[1] / 480
            scale_h = size[0] / 272
            for i in range(len(b_points)):
                y1 = int(np.round(b_points[i][0] * scale_h))
                x1 = int(np.round(b_points[i][1] * scale_w))
                y2 = int(np.round(b_points[i][2] * scale_h))
                x2 = int(np.round(b_points[i][3] * scale_w))
                if x1 == x2:
                    angle = -np.pi / 2
                else:
                    angle = np.arctan((y1-y2) / (x1-x2))
                (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
                b_points[i] = (y1, x1, y2, x2)

            vis = visulize_mapping(b_points, size[::-1], result)
            vis.save(os.path.join(save_path, img_name).replace(".png", "_dht.png"))

        
def evaluate(exp_name, root_input, threshold):
    net = DGNLNet().cuda()

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                       map_location=lambda storage, loc: storage.cuda(0)))

    net.eval()
    avg_time = 0

    with torch.no_grad():

        img_list = make_dataset(root_input, is_train=False)
        MSE, PSNR, SSIM = 0, 0, 0 	
        total_tp = np.zeros(99)
        total_fp = np.zeros(99)
        total_fn = np.zeros(99)

        for idx, path in enumerate(img_list):
            save_path = os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (exp_name, args['snapshot']))
            check_mkdir(save_path)

            
            img_path, gt_path, data_path = path
            data = np.load(data_path, allow_pickle=True).item()
            gt_coords = data["coords"]
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            img_var = Variable(transform(img).unsqueeze(0)).cuda()

            start_time = time.time()
            
            res, key_points = net(img_var)

            torch.cuda.synchronize()

            avg_time = avg_time + time.time() - start_time
            print('predicting: %d / %d, avg_time: %.5f' % (idx + 1, len(img_list), avg_time/(idx+1)))

            ##################### Dehaze ########################
            result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))
            result.save(os.path.join(save_path, img_path.split("/")[-1]))

            gt = Image.open(gt_path).convert('RGB')
            result1 = np.array(result)
            gt = np.array(gt)
#            print(result.shape, gt.shape)
            MSE += mean_squared_error(gt, result1)
            PSNR += peak_signal_noise_ratio(gt, result1)
            SSIM += structural_similarity(gt, result1, multichannel=True)


            ################### DHT ##########################
            key_points = torch.sigmoid(key_points)
            binary_kmap = key_points.squeeze().cpu().numpy() > threshold
            kmap_label = label(binary_kmap, connectivity=1)
            props = regionprops(kmap_label)
            plist = []
            for prop in props:
                plist.append(prop.centroid)
            size = (272, 480)
            b_points = reverse_mapping(plist, numAngle=100, numRho=100, size=(272, 480))

            for i in range(1, 100):
                tp, fp, fn = caculate_tp_fp_fn(b_points, gt_coords, thresh=i*0.01)
                total_tp[i-1] += tp
                total_fp[i-1] += fp
                total_fn[i-1] += fn
                
            
            scale_w = size[1] / 480
            scale_h = size[0] / 272
            for i in range(len(b_points)):
                y1 = int(np.round(b_points[i][0] * scale_h))
                x1 = int(np.round(b_points[i][1] * scale_w))
                y2 = int(np.round(b_points[i][2] * scale_h))
                x2 = int(np.round(b_points[i][3] * scale_w))
                if x1 == x2:
                    angle = -np.pi / 2
                else:
                    angle = np.arctan((y1-y2) / (x1-x2))
                (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
                b_points[i] = (y1, x1, y2, x2)           
            vis = visulize_mapping(b_points, size[::-1], result)
            vis.save(os.path.join(save_path, img_path.split("/")[-1].replace(".png", "_dht.png")))
            
        
        total_recall = total_tp / (total_tp + total_fn + 1e-8)
        total_precision = total_tp / (total_tp + total_fp + 1e-8)
        f = 2 * total_recall * total_precision / (total_recall + total_precision + 1e-8)
        
        print('%s_%s:' % (exp_name, args['snapshot']))
        print("MSE: " + str(MSE / len(img_list)))
        print("PSNR: " + str(PSNR / len(img_list)))
        print("SSIM: " + str(SSIM / len(img_list)))
        print('total_precison: ' + str(total_precision.mean()))
        print('total_recall" '+ str(total_recall.mean()))
        print('f-measure@0.95: ' + str(f[95 - 1]))

        f = open("result.txt", "a")
        f.write('%s_%s:\n' % (exp_name, args['snapshot']))
        f.write("MSE: " + str(MSE / len(img_list)) + "\n")
        f.write("PSNR: " + str(PSNR / len(img_list)) + "\n")
        f.write("SSIM: " + str(SSIM / len(img_list)) + "\n")
        f.write("total_precison: " + str(total_precision.mean()) + "\n")
        f.write("total_recall: "+ str(total_recall.mean()) + "\n")
        f.write("f-measure@0.95: " + str(f[95 - 1]) + "\n")

        f.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='DGNLNet', help='model name')
    parser.add_argument('--root_input', type=str, default='dataset/sewer', help='input dir')
    parser.add_argument('--val', action='store_true', help='if val')
    parser.add_argument('--threshold', type=float, default=0.01, help='for dht')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    print(opt.val, opt.exp_name)
    if opt.val:
        evaluate(opt.exp_name, opt.root_input, opt.threshold)
    else:
        predict(opt.exp_name, opt.root_input, opt.threshold)
