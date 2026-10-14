ROOT_FOLDER=${ROOT_FOLDER:-/path/to/downloaded_data}
OUTPUT_FOLDER=output
N_JOBS=32
IMAGES="brain_t1c brain_t1n brain_t2w brain_t2f"

for tp_id in preop postop
do
    if [ $tp_id == "preop" ]; then
        INPUT_FOLDER=${PREOP_INPUT_FOLDER:-$ROOT_FOLDER/preop_scans_coreg}
        MASK_FOLDER=${PREOP_MASK_FOLDER:-$ROOT_FOLDER/preop_masks}
    else
        INPUT_FOLDER=${POSTOP_INPUT_FOLDER:-$ROOT_FOLDER/postop_scans_coreg}
        MASK_FOLDER=${POSTOP_MASK_FOLDER:-$ROOT_FOLDER/postop_masks}
    fi
    for image in $IMAGES
    do 
        out_path=$OUTPUT_FOLDER/statistics/$tp_id.$image.csv
        if [ -f $out_path ]; then
            echo "Statistics already calculated for $tp_id $image"
            continue
        fi
        uv run python \
            -m src.extract_statistics \
            --input_folder $INPUT_FOLDER \
            --identifier_pattern '(?<=/)[0-9]+(?=/)' \
            --image_pattern ".*$image.*" \
            --output_path $OUTPUT_FOLDER/statistics/$tp_id.$image.csv
    done

    for image in $IMAGES
    do 
        out_path=$OUTPUT_FOLDER/features/$tp_id.$image.csv
        if [ -f $out_path ]; then
            echo "Features already calculated for $tp_id $image"
            continue
        fi
        uv run python \
            -m src.extract_features \
            --input_folder $INPUT_FOLDER \
            --mask_folder $MASK_FOLDER \
            --identifier_pattern '(?<=/)[0-9]+(?=/)' \
            --image_pattern ".*$image.*" \
            --mask_pattern '.*tumor_mask.*' \
            --config_path radiomics/radiomicsConfig.yaml  \
            --output_path $OUTPUT_FOLDER/features/$tp_id.$image.csv \
            --n_jobs $N_JOBS
    done

    uv run python -m src.merge_datasets \
        --input_csvs $(for image in $IMAGES; do echo $OUTPUT_FOLDER/features/$tp_id.$image.csv; done) \
        --suffixes $IMAGES \
        --on identifier mask_label mask_path \
        --keep_from_first original_shape_Elongation original_shape_Flatness original_shape_LeastAxisLength original_shape_MajorAxisLength original_shape_Maximum2DDiameterColumn original_shape_Maximum2DDiameterRow original_shape_Maximum2DDiameterSlice original_shape_Maximum3DDiameter original_shape_MeshVolume original_shape_MinorAxisLength original_shape_Sphericity original_shape_SurfaceArea original_shape_SurfaceVolumeRatio original_shape_VoxelVolume label_sum \
        --output_path $OUTPUT_FOLDER/features/$tp_id.csv \
        --reorder_columns identifier mask_label mask_path label_sum image_path.brain_t1c image_path.brain_t1n image_path.brain_t2w image_path.brain_t2f error.brain_t1c error.brain_t1n error.brain_t2w error.brain_t2f 
done
