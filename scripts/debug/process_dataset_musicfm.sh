layer_id=$1
num=16

python -m src.feature_extraction.process_datasets \
  --dataset slakh \
  --root dataset/slakh2100_flac_redux/ \
  --label-root dataset/slakh2100_flac_redux/ \
  --out-h5 sae-data/slakh2100_train_$num\/musicfm/slakh2100_train_$num\_musicfm_30s_layer_$layer_id.h5 \
  --split train --rep musicfm_layer \
  --musicfm-layer $layer_id \
  --musicfm-model-id tky823/MusicFM \
  --mode mix \
  --duration 30 \
  --dtype float16 \
  --compression lzf \
  --hop 10 \
  --track-id-files dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_1.txt dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_2.txt
