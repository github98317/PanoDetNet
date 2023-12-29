import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import torch
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from utils.loss import SigmoidBin

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return (box, score)


# class ASFF_Detect(nn.Module):  # add ASFFV5 layer and Rfb
#     stride = None  # strides computed during build
#     onnx_dynamic = False  # ONNX export parameter
#     export = False  # export mode
#
#     def __init__(self, nc=3, anchors=(), ch=(), multiplier=0.5, rfb=False, inplace=True):  # detection layer
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         self.l0_fusion = ASFFV5(level=0, multiplier=multiplier, rfb=rfb)
#         self.l1_fusion = ASFFV5(level=1, multiplier=multiplier, rfb=rfb)
#         self.l2_fusion = ASFFV5(level=2, multiplier=multiplier, rfb=rfb)
#         self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
#         self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
#         self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
#         self.inplace = inplace  # use in-place ops (e.g. slice assignment)
#
#     def forward(self, x):
#         z = []  # inference output
#         result = []
#
#         result.append(self.l2_fusion(x))
#         result.append(self.l1_fusion(x))
#         result.append(self.l0_fusion(x))
#         x = result
#         for i in range(self.nl):
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             if not self.training:  # inference
#                 if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                     self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
#
#                 y = x[i].sigmoid()
#                 if self.inplace:
#                     y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
#                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                 else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
#                     xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
#                     xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
#                     wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
#                     y = torch.cat((xy, wh, conf), 4)
#                 z.append(y.view(bs, -1, self.no))
#
#         return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
#
#     def _make_grid(self, nx=20, ny=20, i=0):
#         d = self.anchors[i].device
#         t = self.anchors[i].dtype
#         shape = 1, self.na, ny, nx, 2  # grid shape
#         y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
#         if torch.__version__ >= '1.10.0':  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
#             yv, xv = torch.meshgrid(y, x, indexing='ij')
#         else:
#             yv, xv = torch.meshgrid(y, x)
#         grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
#         anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
#         # print(anchor_grid)
#         return grid, anchor_grid
class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution

        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return (box, score)


class ASFF_Detect(IDetect):
    # ASFF model for improvement
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__(nc, anchors, ch)
        self.nl = len(anchors)
        self.asffs = nn.ModuleList(ASFF(i) for i in range(self.nl))
        self.detect = IDetect.forward

    def forward(self, x):  # x中的特征图从大到小，与ASFF中顺序相反，因此输入前先反向
        x = x[::-1]
        for i in range(self.nl):
            x[i] = self.asffs[i](*x)
        return self.detect(self, x[::-1])


class IKeypoint(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), nkpt=17, ch=(), inplace=True, dw_conv_kpt=False):  # detection layer
        super(IKeypoint, self).__init__()
        self.nc = nc  # number of classes
        self.nkpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.no_det = (nc + 5)  # number of outputs per anchor for box and class
        self.no_kpt = 3 * self.nkpt  ## number of outputs per anchor for keypoints
        self.no = self.no_det + self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)

        if self.nkpt is not None:
            if self.dw_conv_kpt:  # keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                    nn.Sequential(DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)
            else:  # keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            if self.nkpt is None or self.nkpt == 0:
                x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
            else:
                x[i] = torch.cat((self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])), axis=1)

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_det = x[i][..., :6]
            x_kpt = x[i][..., 6:]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    if self.nkpt != 0:
                        x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1, 1, 1, 1, 17)) * \
                                           self.stride[i]  # xy
                        x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1, 1, 1, 1, 17)) * \
                                           self.stride[i]  # xy
                        # x_kpt[..., 0::3] = (x_kpt[..., ::3] + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = (x_kpt[..., 1::3] + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        # print('=============')
                        # print(self.anchor_grid[i].shape)
                        # print(self.anchor_grid[i][...,0].unsqueeze(4).shape)
                        # print(x_kpt[..., 0::3].shape)
                        # x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        # x_kpt[..., 0::3] = (((x_kpt[..., 0::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = (((x_kpt[..., 1::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    if self.nkpt != 0:
                        y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1, 1, 1, 1, self.nkpt))) * \
                                     self.stride[i]  # xy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class IAuxDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IAuxDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])  # output conv
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            x[i + self.nl] = self.m2[i](x[i + self.nl])
            x[i + self.nl] = x[i + self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x[:self.nl])

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IAuxDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return (box, score)


class IBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=(), bin_count=21):  # detection layer
        super(IBin, self).__init__()
        self.nc = nc  # number of classes
        self.bin_count = bin_count

        self.w_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        self.h_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        # classes, x,y,obj
        self.no = nc + 3 + \
                  self.w_bin_sigmoid.get_length() + self.h_bin_sigmoid.get_length()  # w-bce, h-bce
        # + self.x_bin_sigmoid.get_length() + self.y_bin_sigmoid.get_length()

        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):

        # self.x_bin_sigmoid.use_fw_regression = True
        # self.y_bin_sigmoid.use_fw_regression = True
        self.w_bin_sigmoid.use_fw_regression = True
        self.h_bin_sigmoid.use_fw_regression = True

        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                # px = (self.x_bin_sigmoid.forward(y[..., 0:12]) + self.grid[i][..., 0]) * self.stride[i]
                # py = (self.y_bin_sigmoid.forward(y[..., 12:24]) + self.grid[i][..., 1]) * self.stride[i]

                pw = self.w_bin_sigmoid.forward(y[..., 2:24]) * self.anchor_grid[i][..., 0]
                ph = self.h_bin_sigmoid.forward(y[..., 24:46]) * self.anchor_grid[i][..., 1]

                # y[..., 0] = px
                # y[..., 1] = py
                y[..., 2] = pw
                y[..., 3] = ph

                y = torch.cat((y[..., 0:4], y[..., 46:]), dim=-1)

                z.append(y.view(bs, -1, y.shape[-1]))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # if isinstance(m, Detect):
        #     s = 256  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once
        #     # print('Strides: %s' % m.stride.tolist())
        # if isinstance(m, IDetect):
        #     s = 256  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once
        #     # print('Strides: %s' % m.stride.tolist())
        # if isinstance(m, ASFF_Detect):
        #     s = 256  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once
        #     # print('Strides: %s' % m.stride.tolist())
        # if isinstance(m, IAuxDetect):
        #     s = 256  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])  # forward
        #     # print(m.stride)
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_aux_biases()  # only run once
        #     # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, Detect):  ####### 修改 ########
            s = 256  # 2x min stride
            self.model.to(torch.device('cuda'))
            m.stride = torch.tensor([s / x.shape[-2] for x in
                                     self.forward(
                                         torch.zeros(1, ch, s, s).to(torch.device('cuda')))]).cpu()  # forward
            self.model.cpu()
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IDetect):  ####### 修改 ########
            s = 256  # 2x min stride
            self.model.to(torch.device('cuda'))
            m.stride = torch.tensor([s / x.shape[-2] for x in
                                     self.forward(
                                         torch.zeros(1, ch, s, s).to(torch.device('cuda')))]).cpu()  # forward
            self.model.cpu()
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IAuxDetect):  ####### 修改 ########
            s = 256  # 2x min stride
            self.model.to(torch.device('cuda'))
            m.stride = torch.tensor([s / x.shape[-2] for x in
                                     self.forward(torch.zeros(1, ch, s, s).to(torch.device('cuda')))[
                                     :4]]).cpu()  # forward
            self.model.cpu()
            # print(m.stride)
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_aux_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IBin):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases_bin()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IKeypoint):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases_kpt()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced = False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) or isinstance(m,
                                                                                                              IKeypoint) or isinstance(
                        m,
                        ASFF_Detect):
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin, ASFF_Detect))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Bin() module
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            old = b[:, (0, 1, 2, bc + 3)].data
            obj_idx = 2 * bc + 4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, (obj_idx + 1):].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(
                cf / cf.sum())  # cls
            b[:, (0, 1, 2, bc + 3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                # print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
                # m.switch_to_deploy()
            elif isinstance(m, RepConv_OREPA):
                # print(f" switch_to_deploy")
                m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect, IAuxDetect, ASFF_Detect)):
                m.fuse()
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC, SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv,
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                 Res, ResCSPA, ResCSPB, ResCSPC,
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC,
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC, ELAN, ELAN_H, ELAN_RFB_s, SPPCSPC2,
                 ##########修改########
                 Conv_ATT, SPPCSPC_ATT, DSConv2D, SAConv2d, CoordConv, MobileNet_Block, conv_bn_hswish, NCB,
                 FasterNetBlock, FasterNet_PConv, PConv, Conv11, Conv_1331,SPPCSPC_RFB, SPPCSPC_RFB2, SPPCSPC_RFB3,
                 CNeB, RepVGGBlock, FasterNet_PConv_ATT, EVCBlock, RFBNet, GSConv, VoVGSCSP, VoVGSCSPC,RFCAConv,
                 RFB_s, SPPCSPC_RFB4, SPPCSPC_RFB5, GCTConv, ConvNeXtBlock, CAConv, CAMConv, CBAMConv, DCNv2Conv, DCNv2,DCNV3_YoLo]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC, ELAN, ELAN_RFB_s, SPPCSPC2, SPPCSPC_RFB, SPPCSPC_RFB2, SPPCSPC_RFB3,
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,SPPCSPC_RFB4,SPPCSPC_RFB5,
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                     ResCSPA, ResCSPB, ResCSPC,
                     RepResCSPA, RepResCSPB, RepResCSPC,
                     ResXCSPA, ResXCSPB, ResXCSPC,
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC, RFBNet, RFB_s]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is space_to_depth:
            c2 = 4 * ch[f]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Concat_r:
            c2 = sum([ch[x] for x in f])
        elif m is Concat_r_ATT:
            c2 = sum([ch[x] for x in f])
            args = [c2, *args]
        elif m is Concat_ATT:  ##########修改########
            c2 = sum([ch[x] for x in f])
            args = [c2, *args]
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Shortcut_ATT:
            c2 = ch[f[0]]
            args = [c2, *args]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint, ASFF_Detect]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is CARAFE:  ##########修改########
            c2 = ch[f]
            args = [c2, *args]

        elif m is DownSampling:
            c1 = ch[f]
            c2 = 2 * args[0]
            args = [c1, args[0]]
        elif m is DownSampling_H:
            c1 = ch[f[0]]
            c2 = 2 * args[0] + ch[f[1]]
            args = [c1, args[0]]
        elif m is GAM_Attention:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
        elif m is CrissCross:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, *args[1:]]
        elif m is CNeB:  ##########修改########
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m is CNeB:
                args.insert(2, n)
                n = 1
        elif m is RepVGGBlock:  ##########修改########
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m is EVCBlock:
            c2 = ch[f]
            args = [c2, c2]
        # elif m is ASFFV5:
        #     c2 = args[1]
        elif m in {ASFF_2, ASFF_3}:
            c2 = args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args[0] = c2
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    yamldir = ['cfg/addmodel/MobileNetV3/yolov7_MobileNet.yaml',
               'cfg/addmodel/ASFFDetect/yolov7_ASFF_Detect.yaml',
               'cfg/addmodel/MobileNetV3/yolov7.yaml',
               'cfg/addmodel/ConvNeXt/yolov7.yaml',
               'cfg/addmodel/RepVGG/yolov7_RepVGG.yaml',
               'cfg/addmodel/ConvNeXt/yolov7_ConvNeXt.yaml',
               'cfg/addmodel/Att/v7/yolov7_c71_Att.yaml',
               'cfg/addmodel/Att/v7/yolov7_Concat_Att.yaml',
               'cfg/training/yolov7.yaml',
               # 'cfg/addmodel/NextVit/yolov7_NextVit2.yaml',
               'cfg/addmodel/FastNet/yolov7_FastNet.yaml',
               'cfg/addmodel/NextVit/yolov7_NextVit_Conv_Concat_all.yaml',
               'cfg/addmodel/NextVit/yolov7_NextVit_Conv.yaml',
               # 'cfg/addmodel/NextVit/yolov7_NextVit_New.yaml',
               # 'cfg/addmodel/FastNet/yolov7_FastNet_Pconv_ATT.yaml',
               # 'cfg/addmodel/NextVit/yolov7_NextVit_Conv.yaml',
               # 'cfg/addmodel/FastNet/yolov7_FastNet_Pconv_ALL.yaml',
               # 'cfg/addmodel/FastNet/yolov7_FastNet_Pconv_Gai.yaml',
               # 'cfg/addmodel/NextVit/yolov7_NextVit_PConv.yaml',
               # 'cfg/addmodel/SPD_Conv/yolov7_SPD_Conv.yaml',
               # 'cfg/addmodel/CrissCross/yolov7_CrissCross.yaml',
               # 'cfg/addmodel/RFBNet/yolov7_RFBNet.yaml',
               'cfg/addmodel/NextVit/yolov7_NextVit_Conv_SlimNECK.yaml',
               'cfg/training/yolov7-tiny.yaml',
               'cfg/training/yolov7-tiny-AFPN(hight-level做两次).yaml',
               'cfg/training/yolov7-tiny-AFPN(low-level做两次).yaml',
               'cfg/training/yolov7_afpn.yaml',
               # 'cfg/addmodel/NextVit/yolov7_NextVit3.yaml',
               'cfg/addmodel/NextVit/yolov7_NextVit3.yaml',
               # 'cfg/training/yolov7_gai.yaml',
               'cfg/addmodel/FastNet/yolov7_FastNet_Pconv.yaml',
               'cfg/addmodel/NextVit/yolov7_NextVit3.yaml',
               'cfg/addmodel/NextVit/yolov7_fasternetBlock.yaml',
               # 'cfg/addmodel/RFBNet/yolov7_RFBNet.yaml',
               # 'cfg/addmodel/NextVit/yolov7_NextVit_Conv_Gai.yaml',
               'cfg/addmodel/NextVit/yolov7_fasternetBlock_F.yaml',
               'cfg/addmodel/NextVit/yolov7_NextVit.yaml',
               'cfg/training/yolov7_Elan_gai4.yaml',
               'cfg/training/yolov7.yaml',
               # 'cfg/addmodel/RFAConv/yolov7_CAConv.yaml',
               # 'cfg/addmodel/RFAConv/yolov7_CAMConv.yaml',
               # 'cfg/addmodel/RFAConv/yolov7_CBAMConv.yaml',
               # 'cfg/training/yolov7_DCNv2.yaml',
               # 'cfg/training/yolov7_DWConv.yaml',
               'cfg/training/yolov7_sppcspc_RFB5.yaml',
               'cfg/addmodel/RFAConv/yolov7_CAConv3.yaml',
               'cfg/addmodel/RFAConv/yolov7_ELAN_CAConv(SiLU)_SPPCSPC_RFB5.yaml',
               'cfg/addmodel/RFAConv/yolov7_ELAN_CAConv_SPPCSPC_RFB5.yaml',
               'cfg/addmodel/DCNv3/yolov7_DCNv3.yaml',
               'cfg/addmodel/RFAConv/yolov7_CAConv.yaml',
               'cfg/addmodel/DCNv3/yolov7_DCNv3.yaml',
               'cfg/addmodel/RFAConv/yolov7_CAConv.yaml'
               # 'cfg/training/yolov7_3.yaml',
               ]
    parser = argparse.ArgumentParser()
    # parser. add_argument('--cfg', type=str, default='../cfg/addmodel/Att/v7/yolov7_SPPCSPC_Concat_Att.yaml', help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='../cfg/addmodel/ASFFDetect/yolov7_ASFF_Detect.yaml',
    parser.add_argument('--cfg', type=str, default='../' + yamldir[-1], help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='../cfg/training/yolov7_gai.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', default=True, help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    if opt.profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
