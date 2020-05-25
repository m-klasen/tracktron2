# -*- coding = utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN

def add_track_config(cfg: CN):
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.MODEL.TRACK_ON = True

    _C.MODEL.TRACK_HEAD = CN()
    _C.MODEL.TRACK_HEAD.NAME = ""