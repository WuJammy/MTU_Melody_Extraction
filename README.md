# MTU: A Novel Melody Extraction Method Using Swin U-Net Enhanced with Mamba Blocks
## Abstract
MTU is a novel melody extraction method tailored
for polyphonic audio, addressing both vocal and non-vocal music
tracks. Its architecture integrates Swin U-Net with Mamba
blocks, enhancing feature extraction and achieving multi-scale
representation of intricate polyphonic music signals. A key
innovation of MTU is the adoption of multiple patch sequences
with varied orders within Mamba blocks, significantly improving melody extraction accuracy. 
Furthermore, a preprocessing technique involving a Gaussian filter smooths the target salience map,refining melody labeling on the spectrum and boosting training effectiveness. 
Extensive evaluations on benchmark datasets,including ADC2004, MIREX05, and MedleyDB, demonstrate
MTU’s exceptional performance across crucial metrics such as
pitch and chroma accuracy, positioning it as a leading solution
for melody extraction in complex polyphonic scenarios.


## Environment SetUp
- Python Version: 3.8.10
- PyTorch Version: 1.13.1
  CUDA Version 11.6
  ```
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  ```
  CUDA Version 11.7
  ```
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  ```
- Download MeldeyDB Package
  ```
  git clone https://github.com/marl/medleydb.git
  cd medleydb
  pip install .
  pip install -r requirements.txt
  ```
  Set MEDLEYDB_PATH to the local directory of MedleyDB
  ```
  export MEDLEYDB_PATH="MedleyDB Dataset Path"
  ```
  To make it permanent, add the above line ```~/.bashrc```.
- Librosa
  ```
  pip install librosa
  ```
- Mamba
  ```
  pip install causal-conv1d>=1.4.0
  pip install mamba-ssm
  ```
 


 python train_umamba_transformer.py --max_epochs 200 --batch_size 32 --output_dir_name '/home/wujammy/melody_extraction_model'

 python evaluation.py --model_type mamba_transformer_unet --model_path '/home/wujammy/melody_extraction_model/model_mtu_200_batch32_c2_02bce08dice_guassionkernel15_sigmaX05_sigmaY2_eachcolumn_lr0001_3adamW0001_tt.pth' --output_dir_name '/home/wujammy/meldoy_extraction_result' --accuracy_file_name 'accuracy_mtu_kernel15t.csv'
