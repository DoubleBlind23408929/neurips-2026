feature=$1
lid=$2
num=16
m=$3

python -m src.linear_probe.train \
  --h5 sae-data/slakh2100_train_16/$feature\/slakh2100_train_$num\_$feature\_30s_layer_$lid\.h5 \
  --mode $feature\-$lid \
  --task key_relative \
  --labelTag $m\_frame \
  --logDir store/linear_key_$m\_new \
  --runName key_$m\_$feature\-$lid \
  --train-track-ids dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_1.txt \
  --val-track-ids dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_2.txt \
  --batchSize 32 --lr 1e-3 --maxEpochs 80 --top-k 3
