from cProfile import label
from pyexpat import model
import re
import time
from turtle import st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import argparse
import random
from requests import get
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchaudio
from tqdm import tqdm
import csv
import os
from statistics import mean
import torch.nn.functional as F
from mamba_transformer_unet.networks.vision_mamba_transformer import MambaTransformerUnet as ViT_seg 
from mamba_transformer_unet.config import get_train_config
from utils import get_medleydb_audio_names, Audio_dataset, DiceLoss, get_medleydb_vocal_audio_names, get_mir1k_audio_names, get_medleydb_train_audio_names,get_medleydb_validation_audio_names,get_medleydb_vocal_audio_names, get_medleydb_audio_names,get_orchset_name
import torchsummary
from torch.utils.data import ConcatDataset
from pytorch_metric_learning import miners, losses
import mir_eval
from torchstat import stat
from fvcore.nn import FlopCountAnalysis

class OA_Loss(nn.Module):
    def __init__(self):
        super(OA_Loss, self).__init__()

    def forward(self, pred, target):
        pred_list = []
        target_list = []

        pred = pred.squeeze(3)
        target = target.squeeze(3)

        for i in range(pred.size(0)):
            pred_list.append(torch.argmax(pred[i], dim=0))
            target_list.append(torch.argmax(target[i], dim=0))

        pred_list = torch.cat(pred_list, dim=0)
        target_list = torch.cat(target_list, dim=0)

        loss = (F.mse_loss(pred_list.float().requires_grad_(True), target_list.float().requires_grad_(True)))

        return loss

class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=40.0, base=0.5, margin=0.1):
        super(MultiSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.margin = margin

    def forward(self, outputs, labels):
        batch_size, height, width, channel = outputs.size()

        # 將 outputs 和 labels 重塑為二維張量以進行計算
        outputs = outputs.reshape(batch_size, -1)
        labels = labels.reshape(batch_size, -1)

        # 計算余弦相似度
        sim_matrix = F.cosine_similarity(outputs.unsqueeze(1), outputs.unsqueeze(2), dim=3)
        sim_matrix = torch.clamp(sim_matrix, min=1e-6, max=1-1e-6)

        # 獲取正負樣本的 mask
        mask_positive = labels.unsqueeze(1) == labels.unsqueeze(2)
        mask_negative = ~mask_positive

        # 計算正樣本的损失
        pos_sim = sim_matrix[mask_positive].reshape(batch_size, -1)
        pos_sim = torch.sum(torch.exp(-self.alpha * (pos_sim - self.base)), dim=1)

        # 計算負樣本的损失
        neg_sim = sim_matrix[mask_negative].reshape(batch_size, -1)
        neg_sim = torch.sum(torch.exp(self.beta * (neg_sim - self.base)), dim=1)

        # 計算最終損失
        loss = torch.mean(torch.log(1 + pos_sim) + torch.log(1 + neg_sim))
        return loss

class MSE_Position_Loss(nn.Module):
    def __init__(self):
        super(MSE_Position_Loss, self).__init__()

    def forward(self, pred, target):
        pred_list = []
        target_list = []

        pred = pred.squeeze(3)
        target = target.squeeze(3)

        for i in range(pred.size(0)):
            pred_list.append(torch.argmax(pred[i], dim=0))
            target_list.append(torch.argmax(target[i], dim=0))

        pred_list = torch.cat(pred_list, dim=0)
        target_list = torch.cat(target_list, dim=0)

        loss = (F.mse_loss(pred_list.float().requires_grad_(True), target_list.float().requires_grad_(True)))

        return loss

def train_mamba_transformer_unet(config, save=True):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    if config.TRAIN.BATCH_SIZE != 24 and config.TRAIN.BATCH_SIZE % 6 == 0:
        config.TRAIN.BASE_LR *= config.TRAIN.BATCH_SIZE / 24

    audio_medleydb_names = get_medleydb_audio_names()
    audio_mir1k_names = get_mir1k_audio_names(mik1k_folder_path='/home/wujammy/MIR-1K/Wavfile')
    orchset_names = get_orchset_name()
    medleydb_train_audio_names = get_medleydb_train_audio_names()
    medleydb_validation_audio_names = get_medleydb_validation_audio_names()
    audio_vocal_medleydb_names = get_medleydb_vocal_audio_names()
    audio_nonvocal_medleydb_names = get_medleydb_audio_names()
    items_to_remove = []


    for name in audio_nonvocal_medleydb_names:
        if name in audio_vocal_medleydb_names:
            items_to_remove.append(name)

    for item in items_to_remove:
        audio_nonvocal_medleydb_names.remove(item)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViT_seg(config, img_size=config.SPECTRUM.SHAPE[0],
                    num_classes=1).to(torch.device(device))
    
   
    # input_tensor = torch.randn(32, 6, 224, 224).to(torch.device(device))
    # flops = FlopCountAnalysis(model, (input_tensor, ))

    # print(f"FLOPs: {flops.total()/1e9:.2f} G")

    # params = sum(p.numel() for p in model.parameters())
    # print(f"Params: {params / 1e6:.2f} M")

    

    #Check use pretrain model or not
    if config.MODEL.PRETRAIN_CKPT is None:
        pass
    elif config.MODEL.PRETRAIN_CKPT != 'mamba/pretrained_ckpt/swin_tiny_patch4_window7_224.pth':
        model.load_state_dict(
            torch.load(os.path.abspath(config.MODEL.PRETRAIN_CKPT), map_location=device))
    else:  
        model.load_from(config)

    # model.load_state_dict(torch.load('/home/wujammy/melody_extraction_model/model_mtu_200_batch32_c2_02bce08dice_mir1k_noguassionkernel13_sigmaX05_sigmaY1_eachcolumn_lr0001_3adamW0001.pth')) 
    
    #Load training data
    orchset_train = Audio_dataset(audio_names=orchset_names, config=config, augmentation=False,dataset_name='orchset')
    medleydb_train_dataset = Audio_dataset(audio_names=medleydb_train_audio_names, config=config, augmentation=False,dataset_name='medleydb')
    medleydb_validation_dataset = Audio_dataset(audio_names=medleydb_validation_audio_names, config=config, augmentation=False,dataset_name='medleydb')
    mir1k_train = Audio_dataset(audio_names=audio_mir1k_names, config=config, augmentation=False,dataset_name='mir1k')
    
    # medleydb_vocal_train = Audio_dataset(audio_names=audio_vocal_medleydb_names, config=config, augmentation=False,dataset_name='medleydb')
    # medleydb_nonvocal_train = Audio_dataset(audio_names=audio_nonvocal_medleydb_names, config=config, augmentation=False,dataset_name='medleydb')

    #Concatenate two datasets
    # dataset_train = ConcatDataset([medleydb_train, mir1k_train])

    dataset_train = ConcatDataset([orchset_train, medleydb_train_dataset, mir1k_train])
    # dataset_train = ConcatDataset([mir1k_train])
    # dataset_train = ConcatDataset([orchset_train, medleydb_train_dataset])
   
    #divide valid and train dataset 
    # train_size = int(0.9 * len(medleydb_train))
    # valid_size = len(medleydb_train) - train_size

    # train_dataset, valid_dataset = torch.utils.data.random_split(medleydb_train, [train_size, valid_size])

    # divide valid and train dataset
   
    def worker_init_fn(worker_id):
        random.seed(config.SEED + worker_id)

    trainloader = DataLoader(dataset_train,
                             batch_size=config.TRAIN.BATCH_SIZE * config.TRAIN.N_GPU,
                             shuffle=True,
                             num_workers=16,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    validloader = DataLoader(medleydb_validation_dataset,
                            batch_size=config.TRAIN.BATCH_SIZE * config.TRAIN.N_GPU,
                            shuffle=True,
                            num_workers=16,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn,
                            persistent_workers=True)

    if config.TRAIN.N_GPU > 1:
        model = nn.DataParallel(model)

    # input_rand = torch.randn(32,6,224,224).to(torch.device(device))

    # flops, params = profile(model, inputs=(input_rand, ))
    # print('FLOPs: %.2f G' % (flops / 1e9))
    # print('Params: %.2f M' % (params / 1e6))
    # import time 
    # torch.randn(32, 6, 224, 224)
    # model.eval()
    # start = time.time()
    # model(torch.randn(32, 6, 224, 224).to(torch.device(device)))
    # end = time.time()
    # batch1_time = (end - start)/32
    # print('Time: %.5f s' % batch1_time)

    model.train()

    class TverskyLoss(nn.Module):
        def __init__(self, alpha=0.5, beta=0.5, eps=1e-7):
            super(TverskyLoss, self).__init__()
            self.alpha = alpha
            self.beta = beta
            self.eps = eps
    
        def forward(self, y_true, y_pred):
            # Flatten tensors
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)
        
            # Calculate true positives, false positives, and false negatives
            tp = torch.sum(y_true * y_pred)
            fp = torch.sum((1 - y_true) * y_pred)
            fn = torch.sum(y_true * (1 - y_pred))
        
            # Calculate Tversky coefficient
            tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        
            # Calculate Tversky loss
            tversky_loss = 1 - tversky
        
            return tversky_loss
    
    class FocalTverskyLoss(nn.Module):
        def __init__(self, alpha=0.7, gamma=0.75, smooth=1):
            super(FocalTverskyLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.smooth = smooth

        def forward(self, y_true, y_pred):
            # Flatten tensors
            y_true_pos = y_true.reshape(-1)
            y_pred_pos = y_pred.reshape(-1)

            # Calculate true positives, false positives, and false negatives
            true_pos = torch.sum(y_true_pos * y_pred_pos)
            false_neg = torch.sum(y_true_pos * (1-y_pred_pos))
            false_pos = torch.sum((1-y_true_pos)*y_pred_pos)

            # Calculate Tversky coefficient
            tversky = (true_pos + self.smooth)/(true_pos + self.alpha*false_neg + (1-self.alpha)*false_pos + self.smooth)

            # Calculate Focal Tversky Loss
            focal_tversky_loss = torch.pow((1-tversky), self.gamma)

            return focal_tversky_loss
        
        # # flops 

    bce_loss = nn.BCELoss()
    cce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(1)
    tversky_loss = FocalTverskyLoss(alpha=0.7, gamma=0.75)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.TRAIN.BASE_LR,
                            weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.BASE_LR)
    iter_num = 0
    max_iterations = config.TRAIN.EPOCHS * len(trainloader)
    epoch_losses = []
    best_OA = 0
    best_epoch = 0
    no_improvement_epochs = 0
    PATIENCE = 5    

    if save:
        if not os.path.exists(config.OUTPUT_DIR_NAME):
            os.makedirs(config.OUTPUT_DIR_NAME)

    tqdm_iterator = tqdm(range(config.TRAIN.EPOCHS), ncols=70, desc='Training')

    for epoch_num in tqdm_iterator:
        batch_losses = []

        epoch_start_time = time.time()

        for sampled_batch in trainloader:
            audio_batch, label_batch = sampled_batch['audio'], sampled_batch['label']
            audio_batch, label_batch = audio_batch.to(torch.device(device)), label_batch.to(torch.device(device))

            audio_batch = audio_batch.contiguous()

            # 把輸入dimension排序成(batch_size, channel, Width, Height)。原始Swin-Unet是輸入(batch_size, channel, Height, Width)才對，但照此排的accuracy會降低
            audio_batch = audio_batch.permute(0, 3, 2, 1)

            audio_batch = audio_batch.float()

            outputs = model(audio_batch).float()
            
            # # 獲取目前分配的記憶體（單位：位元組）
            # allocated_memory = torch.cuda.memory_allocated()
            # # 獲取目前預留的記憶體（單位：位元組）
            # reserved_memory = torch.cuda.memory_reserved()

            # print(f"Allocated Memory: {allocated_memory / 1024**3:.2f} GB")
            # print(f"Reserved Memory: {reserved_memory / 1024**3:.2f} GB")
         

            # 輸出時的dimension是(batch_size, channel, Width, Height)，把它排序回(batch_size, Height, Width, channel)
            outputs = outputs.permute(0, 3, 2, 1)

            # outputs = nn.Sigmoid()(outputs)
            
            label_batch = label_batch.unsqueeze(3)

            loss = 0.2*cce_loss(outputs.float(), label_batch.float())+0.8*dice_loss(outputs.float(), label_batch.float())
            # loss = cce_loss(outputs.float(), label_batch.float())
            
            # outputs = nn.Softmax(dim=1)(outputs)
            # loss =0.2*bce_loss(outputs.float(), label_batch.float())+0.8*dice_loss(outputs.float(), label_batch.float())    
            # loss = tversky_loss(outputs.float(), label_batch.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            tqdm_iterator.set_description(f'Epoch [{epoch_num+1}/{config.TRAIN.EPOCHS}], Loss: {loss.item():.4f}')

            batch_losses.append(loss.item())

        epoch_end_time = time.time()
        minute_epoch_time = (epoch_end_time - epoch_start_time) / 60
        print('layer 1 Time: %.2f min' % minute_epoch_time)

        lr_ = config.TRAIN.BASE_LR * (1.0 - epoch_num / config.TRAIN.EPOCHS)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        
        model.eval()
        with torch.no_grad():
            OA = 0
            total_OA = 0
            VR = 0 
            VFA = 0 
            RPA = 0
            RCA = 0
           
            for valid_sample_batch in validloader:
                valid_audio_batch, valid_label_batch = valid_sample_batch['audio'], valid_sample_batch['label']
                valid_audio_batch, valid_label_batch = valid_audio_batch.to(torch.device(device)), valid_label_batch.to(torch.device(device))

                valid_audio_batch = valid_audio_batch.contiguous()

                # 把輸入dimension排序成(batch_size, channel, Width, Height)。原始Swin-Unet是輸入(batch_size, channel, Height, Width)才對，但照此排的accuracy會降低
                valid_audio_batch = valid_audio_batch.permute(0, 3, 2, 1)

                valid_audio_batch = valid_audio_batch.float()

                valid_outputs = model(valid_audio_batch).float()

                # 輸出時的dimension是(batch_size, channel, Width, Height)，把它排序回(batch_size, Height, Width, channel)
                valid_outputs = valid_outputs.permute(0, 3, 2, 1)
                
                valid_label_batch = valid_label_batch.unsqueeze(3)

                for batch_num in range(valid_outputs.size(0)):
                    est_freq = torch.argmax(valid_outputs[batch_num].squeeze(), dim=0).cpu().detach().numpy()
                    ref_freq = torch.argmax(valid_label_batch[batch_num].squeeze(), dim=0).cpu().detach().numpy()

                    centerfreq = librosa.cqt_frequencies(n_bins=config.SPECTRUM.N_BINS, fmin=librosa.note_to_hz('C2'),bins_per_octave=config.SPECTRUM.BINS_PER_OCTAVE)

                    ref_time = np.arange(0, ref_freq.shape[0]) / 100
                    est_time = np.arange(0, est_freq.shape[0]) /100

                    for i in range(len(est_freq)):
                        if est_freq[i] == 0:
                            est_freq[i] = 0
                        else:
                            est_freq[i] = centerfreq[int(est_freq[i])]

                    for i in range(len(ref_freq)):
                        if ref_freq[i] == 0:
                            ref_freq[i] = 0
                        else:
                            ref_freq[i] = centerfreq[int(ref_freq[i])]

                    eval = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)

                    OA += eval['Overall Accuracy']
                    VR += eval['Voicing Recall'] 
                    VFA += eval['Voicing False Alarm'] 
                    RPA += eval['Raw Pitch Accuracy'] 
                    RCA += eval['Raw Chroma Accuracy'] 
     
                    total_OA += 1

            OA = OA / total_OA
            VR = VR / total_OA
            VFA = VFA / total_OA
            RPA = RPA / total_OA
            RCA = RCA / total_OA

            print('OA: %f, VR: %f, VFA: %f, RPA: %f, RCA: %f' % (OA, VR, VFA, RPA, RCA))

            OA = (OA + VR + (1-VFA) + RPA + RCA) / 5

            #early stopping
            if OA > best_OA:
                best_OA = OA
                best_epoch = epoch_num
                if save:
                    torch.save(model.state_dict(),
                        os.path.join(config.OUTPUT_DIR_NAME, 'model_mtu.pth'))
            print(" patience ",epoch_num-best_epoch)
            if epoch_num - best_epoch >= PATIENCE:
                print('Early stopping at epoch {}'.format(epoch_num))
                break

        model.train()
                    
        average_batch_loss = mean(batch_losses)

        epoch_losses.append(average_batch_loss)

        print(' epoch %d : Avg loss : %f' % (epoch_num + 1, average_batch_loss))

        if save:
            if (epoch_num + 1) % 50 == 0 and (epoch_num + 1) != config.TRAIN.EPOCHS:
                torch.save(model.state_dict(),
                           os.path.join(config.OUTPUT_DIR_NAME, f'epochs{epoch_num + 1}_model.pth'))

    # 儲存
    if save:
        # torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR_NAME, 'model_2dataset_finalepoch_3adam_adamW0001_layer1.pth'))
        plt.plot(epoch_losses)
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        plt.savefig(os.path.join(config.OUTPUT_DIR_NAME, 'loss_plot_con3_06ce04dice_mamba_25_m2.png'))
        with open(os.path.join(config.OUTPUT_DIR_NAME, 'loss_con3_06ce04dice_mamba_25_m2.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epochs', 'train loss'])

            for i in range(len(epoch_losses)):
                writer.writerow([i, epoch_losses[i]])

    print("Training Finished!")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, help='learning rate')
    parser.add_argument('--n_gpu', type=int, help='total gpu')
    parser.add_argument(
        '--pretrain_ckpt',
        type=str,
        help=
        '在訓練前載入的pre-trained model的路徑。若不指定則使用預設 (swinunet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth)'
    )
    parser.add_argument('--output_dir_name', type=str, help='output directory name')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args = parser.parse_args()
 
    config = get_train_config(args)

    train_mamba_transformer_unet(config, save=True)