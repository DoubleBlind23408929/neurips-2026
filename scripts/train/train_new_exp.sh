lid=$1
feature=$2
mode=$feature\-$lid\_1-1
num=16
dataset=slakh2100_train_$num
lr=1e-3
topk=$3
# Folder of trained chord-probe ckpts; the best-scoring one is auto-selected.
probe_ckpt_folder=store/linear_chord_new/chord_$feature\-$lid/ckpts
probe_ckpt=

# Pick the highest-score ckpt in the folder. Filenames encode the validation
# score as "...scoreNN.NNNN.ckpt" (higher is better for the chord probe), so we
# sort numerically on that field and take the top.
if [ -n "$probe_ckpt_folder" ] && [ -d "$probe_ckpt_folder" ]; then
  probe_ckpt=$(ls "$probe_ckpt_folder"/*.ckpt 2>/dev/null \
    | sed -E 's/.*score([0-9.]+)\.ckpt$/\1 &/' \
    | sort -rn | head -n1 | cut -d' ' -f2-)
fi

if [ -n "$probe_ckpt" ]; then
  echo "[train_sae] using chord probe: $probe_ckpt"
  chord_probe_arg="--chord-probe-ckpt $probe_ckpt"
else
  echo "[train_sae] no chord probe found in '$probe_ckpt_folder'; falling back to ring enumeration"
  chord_probe_arg=
fi

echo "?????????????????????"
python -m src.music_sae.train --mode $mode --featureTag $feature\_layer \
  --h5 sae-data/$dataset\/$feature\/$dataset\_$feature\_30s_layer_$lid\.h5 \
  --train-track-ids dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_1.txt \
  --val-track-ids dataset/slakh2100_flac_redux/text/selected_track_ids_train_$num\_2.txt \
  --runName $dataset\_$mode\_$lr\_$topk\_max \
  --topk $topk \
  --batchSize 32 \
  --logDir store/final_new_1-1/ \
  --numWorkers 6 \
  --lr $lr \
  --probe-sae-idx 0 \
  $chord_probe_arg \
  --maxEpochs 400 