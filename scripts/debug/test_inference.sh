layer_id=$2
feature=$1
feature_layer=$3
#giantsteps_key gtzan

    # --$feature\-layer $layer_id \
# 

#  \
#     --muq-model-id OpenMuQ/MuQ-large-msd-iter \
#     --musicfm-model-id tky823/MusicFM \

dataset=fmakv2
python -m src.feature_extraction.process_datasets  \
    --dataset $dataset \
    --root dataset/FMAKv2 \
    --label-root dataset/FMAKv2 \
    --muq-model-id OpenMuQ/MuQ-large-msd-iter \
    --musicfm-model-id tky823/MusicFM \
    --$feature\-layer $layer_id \
    --split test \
    --out-h5 sae-data/$dataset/$feature\/$dataset\_$feature\_30s_layer_$layer_id\.h5 \
    --rep $feature_layer \
    --mode mix \
    --duration 30 \
    --dtype float16 \
    --inference \
    --require-labels


# dataset=slakh
# python -m src.feature_extraction.process_datasets  \
#     --dataset $dataset \
#     --root dataset/slakh2100_flac_redux \
#     --label-root dataset/slakh2100_flac_redux \
#     --muq-model-id OpenMuQ/MuQ-large-msd-iter \
#     --musicfm-model-id tky823/MusicFM \
#     --$feature\-layer $layer_id \
#     --split test \
#     --out-h5 sae-data/$dataset/$feature\/$dataset\_$feature\_30s_layer_$layer_id\.h5 \
#     --rep $feature_layer \
#     --mode mix \
#     --duration 30 \
#     --dtype float16 \
#     --inference \
#     --require-labels

# --$feature\-layer $layer_id \
# dataset=pop909
# python -m src.feature_extraction.process_datasets  \
#     --dataset $dataset \
#     --root dataset/pop909/pop909_audio \
#     --label-root dataset/pop909/POP909/ \
#     --muq-model-id OpenMuQ/MuQ-large-msd-iter \
#     --musicfm-model-id tky823/MusicFM \
#     --split train \
#     --out-h5 sae-data/$dataset\-train/$feature\/$dataset\_$feature\_30s_layer_$layer_id\.h5 \
#     --rep $feature_layer \
#     --mode mix \
#     --duration 30 \
#     --dtype float16 \
#     --inference \
#     --require-labels

# dataset=rwc
# python -m src.feature_extraction.process_datasets  \
#     --dataset $dataset \
#     --root dataset/rwc/ \
#     --label-root dataset/rwc/ \
#     --split test \
#     --out-h5 sae-data/$dataset/$feature\/$dataset\_$feature\_30s_layer_$layer_id\.h5 \
#     --rep $feature_layer \
#     --$feature\-layer $layer_id  \
#     --mode mix \
#     --duration 30 \
#     --dtype float16 \
#     --inference \
#     --require-labels \
#     --muq-model-id OpenMuQ/MuQ-large-msd-iter \
#     --musicfm-model-id tky823/MusicFM

# dataset=giantsteps_key
# python -m src.feature_extraction.process_datasets  \
#     --dataset $dataset \
#     --root dataset/giantsteps-key-dataset/ \
#     --label-root dataset/giantsteps-key-dataset/ \
#     --split test \
#     --out-h5 sae-data/$dataset/$feature\/$dataset\_$feature\_30s_layer_$layer_id\.h5 \
#     --rep $feature_layer \
#     --$feature\-layer $layer_id  \
#     --mode mix \
#     --duration 30 \
#     --dtype float16 \
#     --inference \
#     --require-labels \
#     --muq-model-id OpenMuQ/MuQ-large-msd-iter \
#     --musicfm-model-id tky823/MusicFM


# dataset=gtzan
# python -m src.feature_extraction.process_datasets  \
#   --dataset $dataset \
#   --root dataset/gtzan \
#   --label-root dataset/gtzan/alexanderlerch-gtzan_key-9ef1591/gtzan_key \
#   --split test \
#   --out-h5 sae-data/$dataset/$feature\/$dataset\_$feature\_30s_layer_$layer_id\.h5 \
#   --rep $feature_layer \
#   --$feature\-layer $layer_id \
#   --mode mix \
#   --duration 30 \
#   --dtype float16 \
#   --inference \
#   --require-labels \
#   --muq-model-id OpenMuQ/MuQ-large-msd-iter \
#   --musicfm-model-id tky823/MusicFM




# dataset=pop909_renders
# python -m src.feature_extraction.process_datasets  \
#     --dataset pop909 \
#     --root dataset/pop909/renders \
#     --label-root dataset/pop909/POP909/ \
#     --muq-model-id OpenMuQ/MuQ-large-msd-iter \
#     --musicfm-model-id tky823/MusicFM \
#     --split train \
#     --out-h5 sae-data/$dataset/$feature\/$dataset\_$feature\_30s_layer_$layer_id\.h5 \
#     --rep $feature_layer \
#     --mode mix \
#     --duration 30 \
#     --dtype float16 \
#     --inference \
#     --require-labels







