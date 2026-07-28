layer_id=$1
num=16

python -m src.feature_extraction.process_datasets \
  --dataset slakh \
  --root dataset/slakh2100_flac_redux/ \
  --label-root dataset/slakh2100_flac_redux/ \
  --out-h5 sae-data/slakh2100_train_$num/muq/slakh2100_train_$num\_muq_30s_layer_$layer_id.h5 \
  --split train --rep muq_layer \
  --muq-layer $layer_id \
  --muq-model-id OpenMuQ/MuQ-large-msd-iter \
  --mode mix \
  --duration 30 \
  --dtype float16 \
  --compression lzf \
  --hop 10 \
  --track-id-files dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_1.txt dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_2.txt
