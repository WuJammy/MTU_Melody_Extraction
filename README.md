# MTU: A Novel Melody Extraction Method Using Swin U-Net Enhanced with Mamba Blocks


 python train_umamba_transformer.py --max_epochs 200 --batch_size 32 --output_dir_name '/home/wujammy/melody_extraction_model'

 python evaluation.py --model_type mamba_transformer_unet --model_path '/home/wujammy/melody_extraction_model/model_mtu_200_batch32_c2_02bce08dice_guassionkernel15_sigmaX05_sigmaY2_eachcolumn_lr0001_3adamW0001_tt.pth' --output_dir_name '/home/wujammy/meldoy_extraction_result' --accuracy_file_name 'accuracy_mtu_kernel15t.csv'
