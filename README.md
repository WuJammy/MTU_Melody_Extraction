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
  You can use Miniconda to setup Python environment.  
  Reference : https://docs.anaconda.com/miniconda/install/.
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
  Note : You may find ```yaml.load()``` errors while running the program. This is due to outdated syntax in the MedleyDB source code. Simply replace ```yaml.load(file)``` with ```yaml.load(file, Loader=yaml.FullLoader)```.
- Librosa
  ```
  pip install librosa
  ```
- Mamba
  ```
  pip install causal-conv1d>=1.4.0
  pip install mamba-ssm
  ```
## Dataset 
- MedleyDB : https://medleydb.weebly.com/
- ORCHSET : https://www.upf.edu/web/mtg/orchset
- MIR-1K : http://mirlab.org/dataset/public/
- ADC2004、MIREX05 : http://labrosa.ee.columbia.edu/projects/melody/

## Update Your Training Dataset Path
Training dataset paths can be configured in  ```utils.py```. Update the following: 
 ```
def get_medleydb_train_audio_names():
    medleydb_trian_names_path = 'Your medleydb_train_names.txt Path'

    audio_names = []

    with open(audio_names_non_vocal_path, 'r') as f:
        for line in f:
            audio_names.append(line.strip())

    return audio_names
 ```
 ```
def get_medleydb_validation_audio_names():
     medleydb_valid_names_path = 'Your medleydb_valid_names.txt Path'

    audio_names = []

    with open(audio_names_non_vocal_path, 'r') as f:
        for line in f:
            audio_names.append(line.strip())

    return audio_names
 ```

 ```
def get_orchset_name():

    orchset_path = 'Your Orchset Path'
    
    #get all audio names(in dir)
    audio_names = [os.path.splitext(f)[0] for f in os.listdir(orchset_path) if f.endswith('.wav')]
    
    return audio_names 
 ```
 ```
def get_mir1k_audio_names(mik1k_folder_path):
    mir1k_path = 'Your mir1k Path'
    
    audio_names = [file_name.stem for file_name in Path(mir1k_path).iterdir()]

    return audio_names
 ```
## Training Model
 ```
 python train_umamba_transformer.py --max_epochs 200 --batch_size 32 --output_dir_name 'Your Model Output Path'
```
## Update Your Test Dataset Path
Test dataset paths can be configured in ```evaluation.py```. Update the following: 
 ```
 tests = [ ('Your MIREX05 Path', '.wav', 'REF.txt'),
              ('Your ADC2004 Path', '.wav', 'REF.txt'),
              ('Yor MedleyDb Path(.txt file)','mdb','mdb'),
             ]  
 ```
## Evaluation Test Dataset  
 ```
 python evaluation.py --model_type mamba_transformer_unet --model_path 'Your Model Path(.pth)' --output_dir_name 'Your Evaluation Output Path ' --accuracy_file_name 'Your Evaluation Result Name(.csv)'
 ```
## 1-D Hz to Midi File
 ```
import change_midi
change_midi.change_midifile(Your 1-D Hz Array , 'Your ouput.mid')
 ```
