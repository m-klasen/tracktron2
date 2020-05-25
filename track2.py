# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

TRACK_HEAD_REGISTRY = Registry("TRACK_HEAD")
TRACK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


class BaseTrackRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic.
    """

    @configurable
    def __init__(self, *, vis_period=0):
        """
        NOTE: this interface is experimental.

        Args:
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD}

    def forward(self, x, ref_x, x_n, ref_x_n):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x, ref_x, x_n, ref_x_n)
        if self.training:
            return x
        else:
            mask_rcnn_inference(x, instances)
            return instances


@TRACK_HEAD_REGISTRY.register()
class TrackRCNNHead(BaseTrackRCNNHead):
    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        super().__init__(**kwargs)
        
        self.in_channels = input_shape.channels
        self.with_avg_pool = False
        self.roi_feat_size = 7
        self.match_coeff = None
        self.bbox_dummy_iou = 0
        self.num_fcs = 2
        fc_out_channels = 1024
        self.in_channels = 1024
        self.fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            self.in_channels = (self.in_channels
                          if i == 0 else fc_out_channels)
            fc = nn.Linear( self.in_channels, fc_out_channels)
            self.fcs.append(fc)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.dynamic=True

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)
    
    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy =  torch.ones(bbox_ious.size(0), 1, 
                device=torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta),dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert(len(self.match_coeff) == 3)
            return match_ll + self.match_coeff[0] * \
                torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                + self.match_coeff[2] * label_delta
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x, ref_x, x_n, ref_x_n):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch, 
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching
        assert x_n == ref_x_n
        if self.with_avg_pool:
            x = self.avg_pool(x)
            ref_x = self.avg_pool(ref_x)
        x = x.view(x.size(0), -1)
        ref_x = ref_x.view(ref_x.size(0), -1)
        for idx, fc in enumerate(self.fcs):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        n = x_n
        #print(x,x_n)
        #import pdb; pdb.set_trace()
        #x_split = torch.split(x, x_n, dim=0)
        #ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        match_score = []
        for i in range(n):
          
            prod = torch.mm(torch.unsqueeze(x[i],0), torch.unsqueeze(ref_x[i],1))
            match_score.append(prod)
        #if self.dynamic:
            
            #for prod in prods:
            #    m = prod.size(0)
            #    dummy = torch.zeros( m, 1, device=torch.cuda.current_device())
                
            #prod_ext = torch.cat([dummy, prod], dim=1)
            #match_score.append(prod_ext)
        #else:
        #    dummy = torch.zeros(n, m, device=torch.cuda.current_device())
        #    prods_all = torch.cat(prods, dim=0)
        #    match_score = torch.cat([dummy,prods_all], dim=2)
        return torch.stack(match_score)

    def loss(self,match_score,ids,reduce=True):
        loss = nn.CrossEntropyLoss()
        losses = dict()
        import pdb;pdb.set_trace()
        if self.dynamic:
            n = len(match_score)
            loss_match = 0.
            match_acc = 0.
            n_total = 0
            batch_size = len(ids)
            loss_match = loss(match_score.squeeze(), ids[0])
                #match_acc += accuracy(torch.index_select(score, 0, valid_idx), 
                #                      torch.index_select(cur_ids,0, valid_idx)) * n_valid
            losses['loss_match'] = loss_match / n
            #if n_total > 0:
            #    losses['match_acc'] = match_acc / n_total
        else:
          if match_score is not None:
              valid_idx = torch.nonzero(cur_weights).squeeze()
              losses['loss_match'] = weighted_cross_entropy(
                  match_score, ids, id_weights, reduce=reduce)
              losses['match_acc'] = accuracy(torch.index_select(match_score, 0, valid_idx), 
                                              torch.index_select(ids, 0, valid_idx))
        return losses


def build_track_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.TRACK_HEAD.NAME
    return TRACK_HEAD_REGISTRY.get(name)(cfg, input_shape)