# generate_heatmap_pannels

This git repo is to generate 4-image panel as the one in example.png at the bottom.

There are different branches for different tumor types.

- 2_classes: for tumor types with binary classification, e.g, BRCA, PAAD
- 3_classes_prad: for 3-way classification, especially for PRAD.
- 6_classes_luad: for 6-way classification, especially for LUAD.

The run instructions are the same for all branches.

## Setup Parameters
You need to change the path in the following codes in main.py. The variable names are self-explanatory 

```python
# these folders will be replaced by paramaters
svs_fol = '/data01/shared/hanle/svs_tcga_paad'  # path to the folder that contains the WSIs
cancer_fol = '/data04/shared/hanle/paad_prediction/data/heatmap_txt_190_tcga' # path to folder that contains the prediction-xxx and color-xxx files from the cancer model. This is the output of cancer model, e.g quip_lung_cancer_detection/data/heatmap_txt/
til_fol = '/data04/shared/shahira/TIL_heatmaps/PAAD/vgg_mix_binary/heatmap_txt' # similar to the cancer_fol but this is the prediction-xxx  and color-xxx files from the TIL pipeline
output_pred = '4panel_pngs'   # path to the output, can be anything of your choice

prefix = "prediction-"
wsi_extension = ".svs"
```


## Usage
python main.py N

where N can be -1, 0, 1, or any positive integer.
- 0/1: not using parallel processing
- any number larger than 1, using N cores in parallel processing, limited to the available cores in the system.
- -1: use all available cores in parallel processing, left 2 cores for others


![4-image panel example for Breast Cancer](https://github.com/hansbu/generate_heatmap_pannels/blob/master/example.png)

