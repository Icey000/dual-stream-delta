python main.py \
--SoccerNet_path ./data/caption-2024/ \
--audio_root ./data/SoccerNet/ \
--model_name Dual-QFormer-gpt2 \
--use_dual_stream \
--GPU 0 \
--pool QFormer \
--NMS_threshold 0.7 \
--max_epochs 20 \
--teacher_forcing_ratio 1 \
--batch_size 48 \
--pretrain \
--window_size_caption 30 \
--max_num_worker 2 \
--model_type gpt \
--gpt_type gpt2 \
--weight_decay 0.05 \
--gpt_dropout 0.2 \
--encoder_dropout 0.3 \
--epochs_classify 15 \
--epochs_caption 20


#--teacher_forcing_ratio 0.9 \
#--continue_training \
#--pool TRANS \
#--window_size_caption 45