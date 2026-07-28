feature=$1
lid=$2
num=16

python -m src.linear_probe.train \
  --task chord \
  --logDir store/linear_chord_new \
  --h5 sae-data/slakh2100_train_16/$feature\/slakh2100_train_16_$feature\_30s_layer_$lid\.h5 \
  --mode $feature\-$lid \
  --runName chord_$feature\-$lid \
  --train-track-ids dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_1.txt \
  --val-track-ids dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_2.txt \
  --val-smooth-win 9 \
  --batchSize 32 --lr 1e-3 --maxEpochs 80 --top-k 3
