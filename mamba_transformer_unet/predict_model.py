import librosa
import numpy as np
import librosa.display
import os
import torch
from torch.utils.data import DataLoader

from .networks.vision_mamba_transformer import MambaTransformerUnet as ViT_seg
from utils import segment_audio, Audio_test_dataset  # 因為在melody_extraction資料夾下執行main.py時，python path內就有melody_extraction資料夾的路徑了，所以直接用absolute import


class melody_extraction_mamba_transformer_unet_model:

    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ViT_seg(config, img_size=config.SPECTRUM.SHAPE[0],
                             num_classes=1).to(torch.device(self.device))
        self.model.load_state_dict(
            torch.load(os.path.abspath(config.TEST.MODEL_PATH), map_location=self.device), strict=False)

        self.config = config

    def prepare_dataset(self, audio_segments):
        db_test = Audio_test_dataset(audio_segments, self.config)

        # 這裡的batch_size不要改，否則predict()那邊要改動
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

        return testloader

    def predict(self, audio_path):
        audio_segments = segment_audio(audio_path=audio_path,
                                       padding_audio_path=self.config.PADDING_AUDIO_PATH,
                                       segment_time=self.config.SEGMENT.AUDIO_SEGMENT_TIME,
                                       overlap_time=self.config.SEGMENT.OVERLAP_TIME)

        testloader = self.prepare_dataset(audio_segments)

        self.model.eval()

        prediction = np.zeros(
            (len(audio_segments), self.config.SPECTRUM.SHAPE[0], self.config.SPECTRUM.SHAPE[1]))

        for i, sampled_batch in enumerate(testloader):
            audio = sampled_batch.permute(0, 3, 2, 1)

            audio = audio.to(torch.device(self.device))

            with torch.no_grad():
                outputs = self.model(audio)

                outputs = outputs.permute(0, 3, 2, 1)
                outputs = outputs.squeeze(3).squeeze(0)  # 把batch_size跟channel的維度刪除，只留二維的predict值

                outputs = outputs.cpu()

                prediction[i] = outputs

        return prediction