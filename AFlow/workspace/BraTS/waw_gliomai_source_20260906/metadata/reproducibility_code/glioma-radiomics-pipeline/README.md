# A simple feature extraction pipeline applied to glioma data

This is a simple feature extraction pipeline for most kinds of images requiring feature extraction using SITK-readable images as input. It allows for the specification of patterns which filter images and masks using regular expressions, as well as the specification of patient/study identifiers in a similar manner and of a radiomics config file.

If co-registration/resampling of multimodal imaging is necessary please consult [this more complex pipeline](https://github.com/CCIG-Champalimaud/radiomics-pipeline).

## Usage

*Requires [`uv`](https://docs.astral.sh/uv/)*

```bash
uv run python -m src.extract_features \
    --input_folder /path/to/input/folder \
    --mask_folder /path/to/mask/folder \
    --identifier_pattern "(.+)" \
    --image_pattern "(.+)" \
    --mask_pattern "(.+)" \
    --config_path /path/to/config.yaml \
    --output_path /path/to/output.csv \
    --n_jobs 8
```

`pipeline.sh` is an example wrapper. Set the input and mask folder environment
variables to the locations used after extracting the public archives.

## Output columns

The output format for columns is `<image_transform_or_diagnostics>_<feature_class>_<feature_name>.<image_type>`.

### `image_transform_or_diagnostics`

* `diagnostics` - diagnostics about the image and mask, useful for debugging
* `original` - original image (no transforms)

### `feature_class`

* `shape` - shape features
* `firstorder` - first-order features
* `glcm` - gray-level co-occurrence matrix features
* `glrlm` - gray-level run-length matrix features
* `glszm` - gray-level size-zone matrix features
* `gldm` - gray-level dependence matrix features
* `ngtdm` - neighborhood gray-tone difference matrix features

### `image_type`

* `brain_t1c` - contrast-enhanced brain T1 image
* `brain_t1n` - regular brain T1 image
* `brain_t2w` - brain T2-weighted image
* `brain_t2f` - brain T2-fluid-attenuated (FLAIR) image

### Additional columns

These are specified as `<column_name>.<image_type>`

* error - error message (if any)
* identifier - identifier of the image (i.e. patient ID)
* mask_label - index label of the mask
* image_path - path to the image
* mask_path - path to the mask
* label_sum - sum of the label (size in voxels)
