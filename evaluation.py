'''
計算main.py獲得的melody的準確率
'''
import librosa
import numpy as np
import librosa.display
import pathlib
import cv2
import csv
import os
import re
import copy
import math
import argparse
from statistics import mean
import mir_eval
from pydub import AudioSegment
import random
import torch
import torch.backends.cudnn as cudnn

from mamba_transformer_unet.predict_model import melody_extraction_mamba_transformer_unet_model
from mamba_transformer_unet.config import get_test_config as get_mamba_transformer_unet_test_config
from utils import spectrum_to_pitches


def get_groundtruth(label_path, audio_time):
    groundtruth = np.loadtxt(label_path)

    # groundtruth padding
    padding_time = audio_time / 1000 - groundtruth[-1, 0]
    if padding_time > 0.0:
        interval = groundtruth[1, 0] - groundtruth[0, 0]
        start = groundtruth[-1, 0] + interval
        num = math.floor(padding_time / interval)
        groundtruth = np.row_stack((groundtruth,
                                    np.column_stack((np.linspace(start,
                                                                 start + num * interval,
                                                                 num=num,
                                                                 endpoint=False), np.zeros(num)))))

    return groundtruth


def compute_metrics(timestamps, groundtruth, prediction):
    # 計算Evaluation Metrics
    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(timestamps, groundtruth,
                                                                   timestamps, prediction)

    return np.array([
        mir_eval.melody.voicing_recall(ref_v, est_v),
        mir_eval.melody.voicing_false_alarm(ref_v, est_v),
        mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c),
        mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c),
        mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c)
    ],
                    dtype=float)


def evaluation_audio(model, audio_path, label_path, config):
    spectrums = model.predict(audio_path)

    input_audio_time = len(AudioSegment.from_file(audio_path))

    groundtruth = get_groundtruth(label_path, input_audio_time)

    # concate spectrums
    start_time = int(0)
    stop_time = config.SEGMENT.AUDIO_SEGMENT_TIME
    i = 0
    merge_spectrum = np.zeros((spectrums.shape[1], 0))
    while start_time < input_audio_time:
        if stop_time <= input_audio_time:  # 最後一個segment預設不更新start_index及end_index，以便resize時直接使用上一次的大小
            # 更新start_index及end_index
            start_index = np.searchsorted(groundtruth[:, 0], start_time / 1000)
            end_index = np.searchsorted(groundtruth[:, 0], stop_time / 1000)

        # segment的label範圍為groundtruth[(start_index:end_index), :]

        # resize
        spectrum = cv2.resize(spectrums[i], (end_index - start_index, spectrums.shape[1]))
        i += 1

        if stop_time > input_audio_time:  # 最後一個segment
            # 計算最後一個segment的start_index及end_index
            start_index = np.searchsorted(groundtruth[:, 0], start_time / 1000)
            end_index = len(
                groundtruth[:, 0])  # 如果剛好碰到stop_time跟最後一個ground truth timestamp相同，則也把該timestamp納入

            # 移除padding部分
            spectrum = spectrum[:, :(end_index - start_index)]

        if start_time > 0:  # 非第一個segment
            # 移除overlap部分
            remove_overlap_start_index = np.searchsorted(
                groundtruth[:, 0], (start_time + config.SEGMENT.OVERLAP_TIME) / 1000)
            spectrum = spectrum[:, (remove_overlap_start_index - start_index):]

        # concate
        merge_spectrum = np.concatenate((merge_spectrum, spectrum), axis=1)

        start_time = stop_time - config.SEGMENT.OVERLAP_TIME
        stop_time = start_time + config.SEGMENT.AUDIO_SEGMENT_TIME

    assert merge_spectrum.shape[1] == len(groundtruth[:, 0]), 'prediction與groundtruth長度不同'

    # 取得prediction值
    prediction = np.array(spectrum_to_pitches(merge_spectrum, config))

    return compute_metrics(groundtruth[:, 0], groundtruth[:, 1], prediction)


def evaluation_model(tests, args=None, save=True, model=None):
    '''
    輸入要evaluation的dataset資訊 (test)及args，輸出shape為(len(tests), 5)的numpy ndarray表示所有dataset的各個evaluation metrics
    
    save: 是否儲存evaluation結果
    model: 若有指定則使用指定的model計算，不開啟model檔案
    '''

    evaluation_metrics_matrices = []  # 所有dataset的evaluation結果都儲存在這
    all_test_audio_names = []
    average_evaluation_metrics_matrix = np.zeros((len(tests), 5), dtype=float)

    if model == None:
        if args.model_type == 'swinunet':
            # 1. swinunet
            config = get_swinunet_test_config(args)
            cudnn.benchmark = False
            cudnn.deterministic = True
            random.seed(config.SEED)
            np.random.seed(config.SEED)
            torch.manual_seed(config.SEED)
            torch.cuda.manual_seed(config.SEED)
            model = melody_extraction_swinunet_model(config)
        elif args.model_type == 'unet_pytorch':
            # 2. unet_pytorch
            config = get_unet_pytorch_test_config(args)
            model = melody_extraction_unet_pytorch_model(config)
        elif args.model_type == 'unet_tensorflow':
            # 3. unet_tensorflow
            config = get_unet_tensorflow_test_config(args)
            model = melody_extraction_unet_tensorflow_model(config)
        elif args.model_type == 'mamba_unet':
            # 4. mamba_unet
            config = get_mamba_unet_test_config(args)
            model = melody_extraction_mamba_unet_model(config)
        elif args.model_type == 'mamba_transformer_unet':
            # 5. mamba_transformer_unet
            config = get_mamba_transformer_unet_test_config(args)
            model = melody_extraction_mamba_transformer_unet_model(config)
        else:
            NotImplementedError('model_type錯誤')

    for test_cnt, (folder_path, audio_suffix, label_suffix) in enumerate(tests):
        test_audio_names = []

        # 取得所有要測試的檔案的檔名
        path = pathlib.Path(folder_path)
        for element in list(path.glob("*.wav")):
            test_audio_names.append(pathlib.Path(element).stem)

        all_test_audio_names.append(test_audio_names)

        evaluation_metrics_matrices.append(np.zeros((len(test_audio_names), 5), dtype=float))

        for audio_cnt, test_audio_name in enumerate(test_audio_names):
            audio_path = os.path.join(folder_path, f'{test_audio_name}{audio_suffix}')
            label_path = os.path.join(folder_path, f'{test_audio_name}{label_suffix}')

            print(test_audio_name)

            evaluation_metrics_matrices[test_cnt][audio_cnt] = evaluation_audio(
                model=model, audio_path=audio_path, label_path=label_path, config=config)

        average_evaluation_metrics_matrix[test_cnt] = np.mean(evaluation_metrics_matrices[test_cnt],
                                                              axis=0)

    # 寫入csv
    if save:
        os.makedirs(config.OUTPUT_DIR_NAME, exist_ok=True)
        csvfile = open(os.path.join(config.OUTPUT_DIR_NAME, config.TEST.ACCURACY_FILE_NAME),
                       'w',
                       newline='')
        writer = csv.writer(csvfile)

        for test_cnt in range(len(tests)):
            writer.writerow([
                'file name', 'voicing recall', 'voicing false alarm', 'raw pitch accuracy',
                'raw chroma accuracy', 'overall accuracy'
            ])

            for audio_cnt, test_audio_name in enumerate(all_test_audio_names[test_cnt]):
                writer.writerow([
                    test_audio_name,
                    *([f'{x:.3f}' for x in evaluation_metrics_matrices[test_cnt][audio_cnt]])
                ])

            writer.writerow(
                ['Avg', *([f'{x:.3f}' for x in average_evaluation_metrics_matrix[test_cnt]])])

            # 空一行
            writer.writerow([])

        csvfile.close()
        print(f'output accuracy file in {config.OUTPUT_DIR_NAME} folder.')

    return average_evaluation_metrics_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type',
        type=str,
        # required=True,
        choices=['swinunet', 'unet_pytorch', 'unet_tensorflow', 'mamba_unet','mamba_transformer_unet'],
        help='swinunet or unet_pytorch or unet_tensorflow or mamba_unet')
    parser.add_argument(
        '--model_path',
        type=str,
        # required=True,
        help='要用來計算Melody Extraction accuracy的model的路徑')
    parser.add_argument('--output_dir_name', type=str, help='放計算完的accuracy資料的directory名稱')
    parser.add_argument('--accuracy_file_name',
                        type=str,
                        default='accuracy.csv',
                        help='accuracy檔案的檔名 (預設: accuracy.csv)')
    parser.add_argument('--all', action='store_true', help='是否要一次計算完所有model資料夾內的所有模型')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args = parser.parse_args()
    # args.model_type = 'swinunet'
    # args.model_path = 'model/230808 swinunet/model.pth'
    # args.output_dir_name = 'model/230808 swinunet'
    # args.accuracy_file_name = 'orchset_accuracy.csv'


    tests = [('/home/wujammy/adc2004_full_set/', '.wav', 'REF.txt'),
             ('/home/wujammy/mirex05TrainFiles/', '.wav', 'REF.txt'),
             ('/home/wujammy/Orchset/', '.wav', '.mel')]  # 預設音檔和ground truth會放在同一個資料夾內

    arg_dicts = []

    if not args.all:
        arg_dicts.append({
            'model_type': args.model_type,
            'model_path': args.model_path,
            'output_dir_name': args.output_dir_name,
            'accuracy_file_name': args.accuracy_file_name
        })
    else:
        for subdir in pathlib.Path('./model').iterdir():
            if subdir.is_dir():
                tmp = {}

                tmp['output_dir_name'] = str(subdir.relative_to(pathlib.Path('.')))

                tmp['model_type'] = subdir.name.split(' ')[-1]

                if tmp['model_type'] == 'swinunet' or tmp['model_type'] == 'unet_pytorch':
                    for file in subdir.glob('*.pth'):
                        prefix = ''

                        pattern = r"(.+)model\.pth"  # 定義正規表達式模式
                        match = re.match(pattern, file.name)
                        if match:
                            prefix = match.group(1)

                        tmp['model_path'] = str(file.relative_to(pathlib.Path('.')))
                        tmp['accuracy_file_name'] = prefix + 'accuracy.csv'

                        arg_dicts.append(copy.deepcopy(tmp))
                elif tmp['model_type'] == 'unet_tensorflow':
                    for file in subdir.glob('*.h5'):
                        prefix = ''

                        pattern = r"(.+)model\.h5"  # 定義正規表達式模式
                        match = re.match(pattern, file.name)
                        if match:
                            prefix = match.group(1)

                        tmp['model_path'] = str(file.relative_to(pathlib.Path('.')))
                        tmp['accuracy_file_name'] = prefix + 'accuracy.csv'

                        arg_dicts.append(copy.deepcopy(tmp))
                else:
                    NotImplementedError('model_type錯誤')

    for arg_dict in arg_dicts:
        args.model_type = arg_dict['model_type']
        args.model_path = arg_dict['model_path']
        args.output_dir_name = arg_dict['output_dir_name']
        args.accuracy_file_name = arg_dict['accuracy_file_name']
        evaluation_model(tests, args=args, save=True)