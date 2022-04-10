import torch
import torch.nn.functional as F
from torch import nn
import  math

class h_swish(nn.Module):
    def __init__(self, inplace = True):
        super(h_swish,self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        x = x * sigmoid
        return x
        
class CoorAttention(nn.Module):
    def __init__(self,in_channels, out_channels, reduction = 32):
        super(CoorAttention, self).__init__()
        self.poolh = nn.AdaptiveAvgPool2d((None, 1))
        self.poolw = nn.AdaptiveAvgPool2d((1,None))
        middle = max(8, in_channels//reduction)
        self.conv1 = nn.Conv2d(in_channels,middle,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(middle)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(middle,out_channels,kernel_size=1,stride=1,padding=0)
        self.conv_w = nn.Conv2d(middle,out_channels,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x): # [batch_size, c, h, w]
        identity = x
        batch_size, c, h, w = x.size()  # [batch_size, c, h, w]
        # X Avg Pool
        x_h = self.poolh(x)    # [batch_size, c, h, 1]

        #Y Avg Pool
        x_w = self.poolw(x)    # [batch_size, c, 1, w]
        x_w = x_w.permute(0,1,3,2) # [batch_size, c, w, 1]

        #following the paper, cat x_h and x_w in dim = 2，W+H
        # Concat + Conv2d + BatchNorm + Non-linear
        y = torch.cat((x_h, x_w), dim=2)   # [batch_size, c, h+w, 1]
        y = self.act(self.bn1(self.conv1(y)))  # [batch_size, c, h+w, 1]
        # split
        x_h, x_w = torch.split(y, [h,w], dim=2)  # [batch_size, c, h, 1]  and [batch_size, c, w, 1]
        x_w = x_w.permute(0,1,3,2)  
        # Conv2d + Sigmoid
        attention_h = self.sigmoid(self.conv_h(x_h))
        attention_w = self.sigmoid(self.conv_w(x_w))
        # re-weight
        return identity * attention_h * attention_w


class SANL(nn.Module):
    def __init__(self, in_channels):
        super(SANL, self).__init__()

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)



    def forward(self, x, lines_map):
        n, c, h, w = x.size()
        x_down = self.down(x)

		# [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)

        ### appearance relation map
        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
        
		# [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        Ra = F.softmax(torch.bmm(theta, phi), 2)

        ### lines relation map
        lines1 = F.interpolate(lines_map, size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners = True).view(n, 1, int(h / 4)*int(w / 4)).transpose(1,2)
        lines2 = F.interpolate(lines_map, size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners = True).view(n, 1, int(h / 8)*int(w / 8))

        # n, (h / 4) * (w / 4), (h / 8) * (w / 8)
        lines1_expand = lines1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        lines2_expand = lines2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))

        Rd = torch.min(lines1_expand / (lines2_expand + self.eps), lines2_expand / (lines1_expand + self.eps))
        
        # normalization: lines relation map [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        # Rd = Rd / (torch.sum(Rd, 2).view(n, int(h / 4) * int(w / 4), 1) + self.eps)

        Rd = F.softmax(Rd, 2)


        # ### position relation map
        # position_h = torch.Tensor(range(h)).cuda().view(h, 1).expand(h, w)
        # position_w = torch.Tensor(range(w)).cuda().view(1, w).expand(h, w)
		#
        # position_h1 = F.interpolate(position_h.unsqueeze(0).unsqueeze(0), size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners=True).view(1, 1, int(h / 4) * int(w / 4)).transpose(1,2)
        # position_h2 = F.interpolate(position_h.unsqueeze(0).unsqueeze(0), size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners=True).view(1, 1, int(h / 8) * int(w / 8))
        # position_h1_expand = position_h1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # position_h2_expand = position_h2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # h_distance = (position_h1_expand - position_h2_expand).pow(2)
		#
        # position_w1 = F.interpolate(position_w.unsqueeze(0).unsqueeze(0), size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners=True).view(1, 1, int(h / 4) * int(w / 4)).transpose(1, 2)
        # position_w2 = F.interpolate(position_w.unsqueeze(0).unsqueeze(0), size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners=True).view(1, 1, int(h / 8) * int(w / 8))
        # position_w1_expand = position_w1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # position_w2_expand = position_w2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # w_distance = (position_w1_expand - position_w2_expand).pow(2)
		#
        # Rp = 1 / (2 * 3.14159265 * self.sigma_pow2) * torch.exp(-0.5 * (h_distance / self.sigma_pow2 + w_distance / self.sigma_pow2))
		#
        # Rp = Rp / (torch.sum(Rp, 2).view(n, int(h / 4) * int(w / 4), 1) + self.eps)


        ### overal relation map
        #S = F.softmax(Ra * Rd * Rp, 2)

        S = F.softmax(Ra * Rd, 2)


        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(S, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners = True)



class NLB(nn.Module):
    def __init__(self, in_channels):
        super(NLB, self).__init__()
        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_down = self.down(x)

        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
        # [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        f = F.softmax(torch.bmm(theta, phi), 2)
        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(f, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation), nn.ReLU()
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.coor = CoorAttention(channels, channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.coor(conv1)
        
        return x + conv2

#x = torch.ones(1,64,2,2)
#a = DilatedResidualBlock(64,1)
#print(a(x).size())

class SpatialRNN(nn.Module):
	"""
	SpatialRNN model for one direction only
	"""
	def __init__(self, alpha = 1.0, channel_num = 1, direction = "right"):
		super(SpatialRNN, self).__init__()
		self.alpha = nn.Parameter(torch.Tensor([alpha] * channel_num))
		self.direction = direction

	def __getitem__(self, item):
		return self.alpha[item]

	def __len__(self):
		return len(self.alpha)


	def forward(self, x):
		"""
		:param x: (N,C,H,W)
		:return:
		"""
		height = x.size(2)
		weight = x.size(3)
		x_out = []

		# from left to right
		if self.direction == "right":
			x_out = [x[:, :, :, 0].clamp(min=0)]

			for i in range(1, weight):
				temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, :, i]).clamp(min=0)
				x_out.append(temp)  # a list of tensor

			return torch.stack(x_out, 3)  # merge into one tensor

		# from right to left
		elif self.direction == "left":
			x_out = [x[:, :, :, -1].clamp(min=0)]

			for i in range(1, weight):
				temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, :, -i - 1]).clamp(min=0)
				x_out.append(temp)

			x_out.reverse()
			return torch.stack(x_out, 3)

		# from up to down
		elif self.direction == "down":
			x_out = [x[:, :, 0, :].clamp(min=0)]

			for i in range(1, height):
				temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, i, :]).clamp(min=0)
				x_out.append(temp)

			return torch.stack(x_out, 2)

		# from down to up
		elif self.direction == "up":
			x_out = [x[:, :, -1, :].clamp(min=0)]

			for i in range(1, height):
				temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, -i - 1, :]).clamp(min=0)
				x_out.append(temp)

			x_out.reverse()
			return torch.stack(x_out, 2)

		else:
			print("Invalid direction in SpatialRNN!")
			return KeyError



class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class NLB(nn.Module):
    def __init__(self, in_channels):
        super(NLB, self).__init__()
        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_down = self.down(x)

        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
        # [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        f = F.softmax(torch.bmm(theta, phi), 2)
        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(f, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)









# class SANLB(nn.Module):
#     def __init__(self, in_channels):
#         super(SANLB, self).__init__()
#
#         self.roll = nn.Conv2d(1, int(in_channels / 2), kernel_size=1)
#         self.ita = nn.Conv2d(1, int(in_channels / 2), kernel_size=1)
#
#         self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
#         self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
#         self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
#
#         self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
#         self.down.weight.data.fill_(1. / 16)
#
#         #self.down_lines = nn.Conv2d(1, 1, kernel_size=4, stride=4, groups=in_channels, bias=False)
#         #self.down_lines.weight.data.fill_(1. / 16)
#
#         self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)
#
#     def forward(self, x, lines):
#         n, c, h, w = x.size()
#         x_down = self.down(x)
#
#         lines_down = F.avg_pool2d(lines, kernel_size=(4,4))
#
#         # [n, (h / 4) * (w / 4), c / 2]
#         #roll = self.roll(lines_down).view(n, int(c / 2), -1).transpose(1, 2)
#         # [n, c / 2, (h / 4) * (w / 4)]
#         #ita = self.ita(lines_down).view(n, int(c / 2), -1)
#         # [n, (h / 4) * (w / 4), (h / 4) * (w / 4)]
#
#         lines_correlation = F.softmax(torch.bmm(lines_down.view(n, 1, -1).transpose(1, 2), lines_down.view(n, 1, -1)), 2)
#
#
#         # [n, (h / 4) * (w / 4), c / 2]
#         theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
#         # [n, c / 2, (h / 8) * (w / 8)]
#         phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
#         # [n, (h / 8) * (w / 8), c / 2]
#         g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
#         # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
#         f_correlation = F.softmax(torch.bmm(theta, phi), 2)
#         # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
#         final_correlation = F.softmax(torch.bmm(lines_correlation, f_correlation), 2)
#
#         # [n, c / 2, h / 4, w / 4]
#         y = torch.bmm(final_correlation, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))
#
#         return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)
