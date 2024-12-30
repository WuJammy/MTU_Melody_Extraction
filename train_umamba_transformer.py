import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import csv
import os
from statistics import mean
from mamba_transformer_unet.networks.vision_mamba_transformer import MambaTransformerUnet as ViT_seg 
from mamba_transformer_unet.config import get_train_config
from utils import get_medleydb_audio_names, Audio_dataset, DiceLoss, get_medleydb_vocal_audio_names, get_mir1k_audio_names, get_medleydb_train_audio_names,get_medleydb_validation_audio_names,get_medleydb_vocal_audio_names, get_medleydb_audio_names,get_orchset_name
from torch.utils.data import ConcatDataset
import mir_eval
from torchstat import stat
from fvcore.nn import FlopCountAnalysis

def train_mamba_transformer_unet(config, save=True):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    if config.TRAIN.BATCH_SIZE != 24 and config.TRAIN.BATCH_SIZE % 6 == 0:
        config.TRAIN.BASE_LR *= config.TRAIN.BATCH_SIZE / 24

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
    

    #Check use pretrain model or not
    if config.MODEL.PRETRAIN_CKPT is None:
        pass
    elif config.MODEL.PRETRAIN_CKPT != 'mamba/pretrained_ckpt/swin_tiny_patch4_window7_224.pth':
        model.load_state_dict(
            torch.load(os.path.abspath(config.MODEL.PRETRAIN_CKPT), map_location=device))
    else:  
        model.load_from(config) 
    
    #Load training data
    orchset_train = Audio_dataset(audio_names=orchset_names, config=config, augmentation=False,dataset_name='orchset')
    medleydb_train_dataset = Audio_dataset(audio_names=medleydb_train_audio_names, config=config, augmentation=False,dataset_name='medleydb')
    medleydb_validation_dataset = Audio_dataset(audio_names=medleydb_validation_audio_names, config=config, augmentation=False,dataset_name='medleydb')
    mir1k_train = Audio_dataset(audio_names=audio_mir1k_names, config=config, augmentation=False,dataset_name='mir1k')

    dataset_train = ConcatDataset([orchset_train, medleydb_train_dataset, mir1k_train])
  
    # valid and train dataset
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

    model.train() 

    cce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.TRAIN.BASE_LR,
                            weight_decay=0.0001)
   
    iter_num = 0
    epoch_losses = []
    best_OA = 0
    best_epoch = 0
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
            

            # 輸出時的dimension是(batch_size, channel, Width, Height)，把它排序回(batch_size, Height, Width, channel)
            outputs = outputs.permute(0, 3, 2, 1)

            # outputs = nn.Sigmoid()(outputs)
            
            label_batch = label_batch.unsqueeze(3)

            loss = 0.2*cce_loss(outputs.float(), label_batch.float())+0.8*dice_loss(outputs.float(), label_batch.float())
            
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