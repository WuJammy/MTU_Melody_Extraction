# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

from os.path import join as pjoin

import torch
import torch.nn as nn

from torch.nn.modules.utils import _pair
from scipy import ndimage
from .mamba_transformer_unet_skip_expand_decoder_sys import MambaTransformerSys

logger = logging.getLogger(__name__)


class MambaTransformerUnet(nn.Module):

    def __init__(self, config, img_size=224, num_classes=1, zero_head=False, vis=False):
        super(MambaTransformerUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.mamba_transformer_unet = MambaTransformerSys(img_size=config.SPECTRUM.SHAPE[0],
                                  patch_size=config.MODEL.MAMBATRANSFORMER.PATCH_SIZE,
                                  in_chans=config.MODEL.MAMBATRANSFORMER.IN_CHANS,
                                  num_classes=self.num_classes,
                                  embed_dim=config.MODEL.MAMBATRANSFORMER.EMBED_DIM,
                                  depths=config.MODEL.MAMBATRANSFORMER.DEPTHS,
                                  decoder_depths=config.MODEL.MAMBATRANSFORMER.DECODER_DEPTHS,
                                  num_heads=config.MODEL.MAMBATRANSFORMER.NUM_HEADS,
                                  patch_norm=config.MODEL.MAMBATRANSFORMER.PATCH_NORM,
                                  use_checkpoint=False)  # 修改為完全不用use_checkpoint這功能

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.mamba_transformer_unet(x)
        return logits

    def load_from(self, config):
        '''
        使用pre-train model。但目前好像無法用此function正常載入訓練好的swinunet model作為pre-train model
        '''

        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")
            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(
                            k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
