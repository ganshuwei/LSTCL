CUDA_VISIBLE_DEVICES=3 python downstream_phase/run_phase_training.py \
--batch_size 8 \
--epochs 50 \
--save_ckpt_freq 10 \
--model  surgformer_HTA \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--finetune  /data/gsw/Code/Surgformer/results/Cholec80/surgformer_HTA_Cholec80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-best.pth \
--data_path /data/gsw/Data/cholec80 \
--eval_data_path /data/gsw/Data/cholec80 \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--eval \
--data_set Cholec80 \
--data_fps 1fps \
--output_dir /data/gsw/Code/Surgformer/results/Cholec80 \
--log_dir /data/gsw/Code/Surgformer/results/Cholec80 \
--num_workers 10 \
--dist_eval \
--enable_deepspeed \
--no_auto_resume \
--cut_black