import os
import math
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.PRETRAIN_CKPT = None
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.25

# Swin Transformer parameters
_C.MODEL.MAMBATRANSFORMER = CN()
_C.MODEL.MAMBATRANSFORMER.PATCH_SIZE = [2,2]
_C.MODEL.MAMBATRANSFORMER.IN_CHANS = 6  # !!!!!!!!!!!!!!!修改!!!!!!!!!!!!!!!，原本值是3，因為U-Net的channel數是6，所以改成6
_C.MODEL.MAMBATRANSFORMER.EMBED_DIM = 96
_C.MODEL.MAMBATRANSFORMER.DEPTHS = [2, 2, 2, 1]
_C.MODEL.MAMBATRANSFORMER.DECODER_DEPTHS = [1, 2, 2, 2]
_C.MODEL.MAMBATRANSFORMER.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.MAMBATRANSFORMER.PATCH_NORM = True
_C.MODEL.MAMBATRANSFORMER.FINAL_UPSAMPLE = "expand_first"

# -----------------------------------------------------------------------------
# Audio segment settings
# -----------------------------------------------------------------------------
_C.SEGMENT = CN()
_C.SEGMENT.AUDIO_SEGMENT_TIME = 1300  # 單位: ms
_C.SEGMENT.OVERLAP_TIME = 300  # 單位: ms

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
# Training dataset
_C.DATASET = CN()
_C.DATASET.SAMPLING_TIME = 0.005804988662131519  # medleydb
# _C.DATASET.SAMPLING_TIME = 0.01  # orchset

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 150
_C.TRAIN.BATCH_SIZE = 24
_C.TRAIN.BASE_LR = 0.0001 # 0.001
_C.TRAIN.N_GPU = 1

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.MODEL_TYPE = None
_C.TEST.MODEL_PATH = None
_C.TEST.AUDIO_PATH = None
_C.TEST.ACCURACY_FILE_NAME = 'accuracy.csv'

# -----------------------------------------------------------------------------
# Audio spectrum settings
# -----------------------------------------------------------------------------
_C.SPECTRUM = CN()
_C.SPECTRUM.N_BINS = 224
_C.SPECTRUM.BINS_PER_OCTAVE = 37
_C.SPECTRUM.HOP_LENGTH = 256
_C.SPECTRUM.SHAPE = (_C.SPECTRUM.N_BINS,
                     math.ceil((_C.SEGMENT.AUDIO_SEGMENT_TIME / 1000) / _C.DATASET.SAMPLING_TIME)
                     )  # 以medleydb為例，unet是(360, 517)，swinunet是(224, 224)

# -----------------------------------------------------------------------------
# MIR-1K dataset settings
# -----------------------------------------------------------------------------
_C.MIR1K = CN()
_C.MIR1K.AUDIO_PATH = os.path.abspath('/home/wujammy/MIR-1K/Wavfile')
_C.MIR1K.LABEL_PATH = os.path.abspath('/home/wujammy/MIR-1K/PitchLabel')

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.PADDING_AUDIO_PATH = os.path.abspath('./padding_audio.wav')
_C.SEED = 1234
_C.OUTPUT_DIR_NAME = 'output'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def get_train_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()

    # _update_config_from_file(config, args.cfg)

    config.defrost()
    if hasattr(args, 'opts') and args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if hasattr(args, 'pretrain_ckpt') and args.pretrain_ckpt:
        config.MODEL.PRETRAIN_CKPT = args.pretrain_ckpt
    if hasattr(args, 'max_epochs') and args.max_epochs:
        config.TRAIN.EPOCHS = args.max_epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if hasattr(args, 'base_lr') and args.base_lr:
        config.TRAIN.BASE_LR = args.base_lr
    if hasattr(args, 'n_gpu') and args.n_gpu:
        config.TRAIN.N_GPU = args.n_gpu
    if hasattr(args, 'output_dir_name') and args.output_dir_name:
        config.OUTPUT_DIR_NAME = args.output_dir_name

    config.freeze()

    return config


def get_test_config(args):
    config = _C.clone()

    # _update_config_from_file(config, args.cfg)

    config.defrost()
    if hasattr(args, 'opts') and args.opts:
        config.merge_from_list(args.opts)

    if hasattr(args, 'model_type') and args.model_type:
        config.TEST.MODEL_TYPE = args.model_type
    if hasattr(args, 'model_path') and args.model_path:
        config.TEST.MODEL_PATH = args.model_path
    if hasattr(args, 'audio_path') and args.audio_path:
        config.TEST.AUDIO_PATH = args.audio_path
    if hasattr(args, 'accuracy_file_name') and args.accuracy_file_name:
        config.TEST.ACCURACY_FILE_NAME = args.accuracy_file_name
    if hasattr(args, 'output_dir_name') and args.output_dir_name:
        config.OUTPUT_DIR_NAME = args.output_dir_name

    config.freeze()

    return config