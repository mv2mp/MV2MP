# MV2MP: Segmentation Free Performance Capture of Humans in Direct Physical Contact from Sparse Multi-Cam Setups

## Instructions for Launching

1. Place one of the models in the smpl_models folder: SMPL_MALE.pkl, SMPL_FEMALE.pkl, or SMPL_NEUTRAL.pkl. (the choice depends on the gender settings in the YAML configuration)
2. Set the path to the data folder:
   
```
   export HI4D_DIR=/path/hi4d/pair21/hug21/ 
```
3. Run the following command:
```
bash run_docker.sh  
```

The data folder should have the following structure:
   - cameras  
   - frames  
   - frames_vis  
   - images  
   - meta.npz  
   - seg  
   - smpl

The model results will be stored in hydra_outputs


## Resulting meshes comparison

### Hi4d "Sidehug"

| Frame | DMC | MultiNeuralBody | Ours | GT |
|------------|--------|--------|--------|--------|
| 10 | ![alt](outputs/pair32_sidehug32_unlit/10/mesh_list/dmc_7_1.gif) |  ![alt](outputs/pair32_sidehug32_unlit/10/mesh_list/real.multinb-val-7_1.gif) | ![alt](outputs/pair32_sidehug32_unlit/10/mesh_list/v2a_7_1.gif)|![alt](outputs/pair32_sidehug32_unlit/10/mesh_list/gt.gif)|
| 40 | ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/dmc_7_1.gif) |  ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/real.multinb-val-7_1.gif) | ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/v2a_7_1.gif)|![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/gt.gif)|
| 80 | ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/dmc_7_1.gif) |  ![alt](outputs/pair32_sidehug32_unlit/80/mesh_list/real.multinb-val-7_1.gif) | ![alt](outputs/pair32_sidehug32_unlit/80/mesh_list/v2a_7_1.gif)|![alt](outputs/pair32_sidehug32_unlit/80/mesh_list/gt.gif)|


### Hi4d "Yoga"

| Frame | DMC | MultiNeuralBody | Ours | GT |
|------------|--------|--------|--------|--------|
| 10 | ![alt](outputs/pair00_yoga00_unlit/10/mesh_list/dmc_7_1.gif) |  ![alt](outputs/pair00_yoga00_unlit/10/mesh_list/real.multinb-val-7_1.gif) | ![alt](outputs/pair00_yoga00_unlit/10/mesh_list/v2a_7_1.gif)|![alt](outputs/pair00_yoga00_unlit/10/mesh_list/gt.gif)|
| 40 | ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/dmc_7_1.gif) |  ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/real.multinb-val-7_1.gif) | ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/v2a_7_1.gif)|![alt](outputs/pair00_yoga00_unlit/40/mesh_list/gt.gif)|
| 80 | ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/dmc_7_1.gif) |  ![alt](outputs/pair00_yoga00_unlit/80/mesh_list/real.multinb-val-7_1.gif) | ![alt](outputs/pair00_yoga00_unlit/80/mesh_list/v2a_7_1.gif)|![alt](outputs/pair00_yoga00_unlit/80/mesh_list/gt.gif)|

### Hi4d "Hug"

| Frame | DMC | MultiNeuralBody | Ours | GT |
|------------|--------|--------|--------|--------|
| 11 | ![alt](outputs/pair21_hug21_unlit/11/mesh_list/dmc_7_1.gif) |  ![alt](outputs/pair21_hug21_unlit/11/mesh_list/real.multinb-val-7_1.gif) | ![alt](outputs/pair21_hug21_unlit/11/mesh_list/v2a_7_1.gif)|![alt](outputs/pair21_hug21_unlit/11/mesh_list/gt.gif)|
| 41 | ![alt](outputs/pair21_hug21_unlit/41/mesh_list/dmc_7_1.gif) |  ![alt](outputs/pair21_hug21_unlit/41/mesh_list/real.multinb-val-7_1.gif) | ![alt](outputs/pair21_hug21_unlit/41/mesh_list/v2a_7_1.gif)|![alt](outputs/pair21_hug21_unlit/41/mesh_list/gt.gif)|
| 81 | ![alt](outputs/pair21_hug21_unlit/41/mesh_list/dmc_7_1.gif) |  ![alt](outputs/pair21_hug21_unlit/81/mesh_list/real.multinb-val-7_1.gif) | ![alt](outputs/pair21_hug21_unlit/81/mesh_list/v2a_7_1.gif)|![alt](outputs/pair21_hug21_unlit/81/mesh_list/gt.gif)|

### CMU panoptic "Haggling"

| Frame | DMC | MultiNeuralBody | Ours |
|------------|--------|--------|--------|
| 9000 | ![alt](outputs/cmu_panoptic_haggling_a2/9000/mesh_list/dmc_8_3.gif) |  ![alt](outputs/cmu_panoptic_haggling_a2/9000/mesh_list/real.multinb-val-8_3.gif) | ![alt](outputs/cmu_panoptic_haggling_a2/9000/mesh_list/v2a_8_3.gif)|
| 9030 | ![alt](outputs/cmu_panoptic_haggling_a2/9030/mesh_list/dmc_8_3.gif) |  ![alt](outputs/cmu_panoptic_haggling_a2/9030/mesh_list/real.multinb-val-8_3.gif) | ![alt](outputs/cmu_panoptic_haggling_a2/9030/mesh_list/v2a_8_3.gif)|
| 9090 | ![alt](outputs/cmu_panoptic_haggling_a2/9090/mesh_list/dmc_8_3.gif) |  ![alt](outputs/cmu_panoptic_haggling_a2/9090/mesh_list/real.multinb-val-8_3.gif) | ![alt](outputs/cmu_panoptic_haggling_a2/9090/mesh_list/v2a_8_3.gif)|


## Resulting renders comparison

### Hi4d "Sidehug"

| Frame | MultiNeuralBody | Ours | GT |
|------------|--------|--------|--------|
| 10 | ![alt](outputs/pair32_sidehug32_unlit/10/images_undistorted/real.multinb-val-7_1_76.png) |  ![alt](outputs/pair32_sidehug32_unlit/10/images_undistorted/render_v2a_7_1_76.png) | ![alt](outputs/pair32_sidehug32_unlit/10/images_undistorted/real_76.png)|
| 40 | ![alt](outputs/pair32_sidehug32_unlit/40/images_undistorted/real.multinb-val-7_1_76.png) |  ![alt](outputs/pair32_sidehug32_unlit/40/images_undistorted/render_v2a_7_1_76.png) | ![alt](outputs/pair32_sidehug32_unlit/40/images_undistorted/real_76.png)|
| 80 | ![alt](outputs/pair32_sidehug32_unlit/80/images_undistorted/real.multinb-val-7_1_76.png) |  ![alt](outputs/pair32_sidehug32_unlit/80/images_undistorted/render_v2a_7_1_76.png) | ![alt](outputs/pair32_sidehug32_unlit/80/images_undistorted/real_76.png)|

### Hi4d "Yoga"

| Frame | MultiNeuralBody | Ours | GT |
|------------|--------|--------|--------|
| 10 | ![alt](outputs/pair00_yoga00_unlit/10/images_undistorted/real.multinb-val-7_1_16.png) |  ![alt](outputs/pair00_yoga00_unlit/10/images_undistorted/render_v2a_7_1_16.png) | ![alt](outputs/pair00_yoga00_unlit/10/images_undistorted/real_16.png)|
| 40 | ![alt](outputs/pair00_yoga00_unlit/40/images_undistorted/real.multinb-val-7_1_16.png) |  ![alt](outputs/pair00_yoga00_unlit/40/images_undistorted/render_v2a_7_1_16.png) | ![alt](outputs/pair00_yoga00_unlit/40/images_undistorted/real_16.png)|
| 80 | ![alt](outputs/pair00_yoga00_unlit/80/images_undistorted/real.multinb-val-7_1_16.png) |  ![alt](outputs/pair00_yoga00_unlit/80/images_undistorted/render_v2a_7_1_16.png) | ![alt](outputs/pair00_yoga00_unlit/80/images_undistorted/real_16.png)|


### Hi4d "Hug"

| Frame | MultiNeuralBody | Ours | GT |
|------------|--------|--------|--------|
| 11 | ![alt](outputs/pair21_hug21_unlit/11/images_undistorted/real.multinb-val-7_1_4.png) |  ![alt](outputs/pair21_hug21_unlit/11/images_undistorted/render_v2a_7_1_4.png) | ![alt](outputs/pair21_hug21_unlit/11/images_undistorted/real_4.png)|
| 41 | ![alt](outputs/pair21_hug21_unlit/41/images_undistorted/real.multinb-val-7_1_4.png) |  ![alt](outputs/pair21_hug21_unlit/41/images_undistorted/render_v2a_7_1_4.png) | ![alt](outputs/pair21_hug21_unlit/41/images_undistorted/real_4.png)|
| 81 | ![alt](outputs/pair21_hug21_unlit/81/images_undistorted/real.multinb-val-7_1_4.png) |  ![alt](outputs/pair21_hug21_unlit/81/images_undistorted/render_v2a_7_1_4.png) | ![alt](outputs/pair21_hug21_unlit/81/images_undistorted/real_4.png)|


### CMU panoptic "Haggling"

| Frame | MultiNeuralBody | Ours | GT |
|------------|--------|--------|--------|
| 9000 | ![alt](outputs/cmu_panoptic_haggling_a2/9000/images_undistorted/real.multinb-val-8_3_00_08.png) |  ![alt](outputs/cmu_panoptic_haggling_a2/9000/images_undistorted/render_v2a_8_3_00_08.png) | ![alt](outputs/cmu_panoptic_haggling_a2/9000/images_undistorted/gt_00_08.png)|
| 9030 | ![alt](outputs/cmu_panoptic_haggling_a2/9030/images_undistorted/real.multinb-val-8_3_00_08.png) |  ![alt](outputs/cmu_panoptic_haggling_a2/9030/images_undistorted/render_v2a_8_3_00_08.png) | ![alt](outputs/cmu_panoptic_haggling_a2/9030/images_undistorted/gt_00_08.png)|
| 9090 | ![alt](outputs/cmu_panoptic_haggling_a2/9090/images_undistorted/real.multinb-val-8_3_00_08.png) |  ![alt](outputs/cmu_panoptic_haggling_a2/9090/images_undistorted/render_v2a_8_3_00_08.png) | ![alt](outputs/cmu_panoptic_haggling_a2/9090/images_undistorted/gt_00_08.png)|

## Cameras sparcity investigation. 



### Hi4d  "Yoga" frame 40
| Cameras | DMC  |  MultiNeuralBody | Ours| GT |
|------------|--------|--------|--------|--------|
|7 | ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/dmc_7_1.gif) |         ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/real.multinb-val-7_1.gif) |         ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/v2a_7_1.gif)          |![alt](outputs/pair00_yoga00_unlit/40/mesh_list/gt.gif)|
|5 | ![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/dmc_5_3.gif)| ![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/real.multinb-val-5_3.gif) |![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/v2a_5_3.gif) | ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/gt.gif)|
|3 | ![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/dmc_3_3.gif)| diverged |                                                                          ![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/v2a_3_3.gif) | ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/gt.gif)|


### Hi4d  "Sidehug" frame 40
| Cameras | DMC  |  MultiNeuralBody | Ours| GT |
|------------|--------|--------|--------|--------|
|7 | ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/dmc_7_1.gif) |         ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/real.multinb-val-7_1.gif) |         ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/v2a_7_1.gif)          |![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/gt.gif)|
|5 | ![alt](ablation_outputs/pair32_sidehug32_unlit/40/mesh_list/dmc_5_3.gif)| ![alt](ablation_outputs/pair32_sidehug32_unlit/40/mesh_list/real.multinb-val-5_3.gif) | ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/v2a_7_1.gif)  | ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/gt.gif)|
|3 | ![alt](ablation_outputs/pair32_sidehug32_unlit/40/mesh_list/dmc_3_3.gif)| diverged |                                                                          ![alt](ablation_outputs/pair32_sidehug32_unlit/40/mesh_list/v2a_3_3.gif) | ![alt](outputs/pair32_sidehug32_unlit/40/mesh_list/gt.gif)|


### Hi4d  "Hug" frame 51
| Cameras | DMC  |  MultiNeuralBody | Ours| GT |
|------------|--------|--------|--------|--------|
|7 | ![alt](outputs/pair21_hug21_unlit/51/mesh_list/dmc_7_1.gif) |         ![alt](outputs/pair21_hug21_unlit/51/mesh_list/real.multinb-val-7_1.gif) |         ![alt](outputs/pair21_hug21_unlit/51/mesh_list/v2a_7_1.gif)          |![alt](outputs/pair21_hug21_unlit/51/mesh_list/gt.gif)|
|5 | ![alt](ablation_outputs/pair21_hug21_unlit/51/mesh_list/dmc_5_3.gif)| ![alt](ablation_outputs/pair21_hug21_unlit/51/mesh_list/real.multinb-val-5_3.gif) |![alt](ablation_outputs/pair21_hug21_unlit/51/mesh_list/v2a_5_3.gif) | ![alt](outputs/pair21_hug21_unlit/51/mesh_list/gt.gif)|
|3 | ![alt](ablation_outputs/pair21_hug21_unlit/51/mesh_list/dmc_3_3.gif)| diverged |                                                                          ![alt](ablation_outputs/pair21_hug21_unlit/51/mesh_list/v2a_3_3.gif) | ![alt](outputs/pair21_hug21_unlit/51/mesh_list/gt.gif)|

## Ablation. Smpl fitting. 

### MESH 
| scene|frame | smpl fitting mesh| no smpl fitting mesh  | GT|
|------------|--------|--------|--------|--------|
|hi4d Yoga|40|![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/v2a_7_1.gif) |![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/v2a_wo_smpl_7_1.gif)| ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/gt.gif)|
|hi4d Yoga|60|![alt](ablation_outputs/pair00_yoga00_unlit/60/mesh_list/v2a_7_1.gif) |![alt](ablation_outputs/pair00_yoga00_unlit/60/mesh_list/v2a_wo_smpl_7_1.gif)| ![alt](outputs/pair00_yoga00_unlit/60/mesh_list/gt.gif)|




### RENDER 
| scene|frame | smpl fitting mesh| no smpl fitting mesh  | GT|
|------------|--------|--------|--------|--------|
|hi4d Yoga|40|![alt](ablation_outputs/pair00_yoga00_unlit/40/images_undistorted/render_v2a_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/40/images_undistorted/render_v2a_wo_smpl_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/40/images_undistorted/real_16.png)|
|hi4d Yoga|60|![alt](ablation_outputs/pair00_yoga00_unlit/60/images_undistorted/render_v2a_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/60/images_undistorted/render_v2a_wo_smpl_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/60/images_undistorted/real_16.png)|


## Ablation. Background. 

### MESH 
| scene|frame | background| no background  | GT|
|------------|--------|--------|--------|--------|
|hi4d Yoga|40|![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/v2a_7_1.gif) |![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/v2a_wo_bg_7_1.gif)| ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/gt.gif)|
|hi4d Yoga|60|![alt](ablation_outputs/pair00_yoga00_unlit/60/mesh_list/v2a_7_1.gif) |![alt](ablation_outputs/pair00_yoga00_unlit/60/mesh_list/v2a_wo_bg_7_1.gif)| ![alt](outputs/pair00_yoga00_unlit/60/mesh_list/gt.gif)|


### RENDER 
| scene|frame | background| no background  | GT|
|------------|--------|--------|--------|--------|
|hi4d Yoga|40|![alt](ablation_outputs/pair00_yoga00_unlit/40/images_undistorted/render_v2a_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/40/images_undistorted/render_v2a_wo_bg_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/40/images_undistorted/real_16.png)|
|hi4d Yoga|60|![alt](ablation_outputs/pair00_yoga00_unlit/60/images_undistorted/render_v2a_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/60/images_undistorted/render_v2a_wo_bg_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/60/images_undistorted/real_16.png)|

## Ablation. New compositional rendering. 

### MESH 
| scene|frame | new comp rend| no new comp rend  | GT|
|------------|--------|--------|--------|--------|
|hi4d Yoga|40|![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/v2a_7_1.gif) |![alt](ablation_outputs/pair00_yoga00_unlit/40/mesh_list/v2a_wo_comp_7_1.gif)| ![alt](outputs/pair00_yoga00_unlit/40/mesh_list/gt.gif)|
|hi4d Yoga|60|![alt](ablation_outputs/pair00_yoga00_unlit/60/mesh_list/v2a_7_1.gif) |![alt](ablation_outputs/pair00_yoga00_unlit/60/mesh_list/v2a_wo_comp_7_1.gif)| ![alt](outputs/pair00_yoga00_unlit/60/mesh_list/gt.gif)|


### RENDER 
| scene|frame |new comp rend| no new comp rend  | GT|
|------------|--------|--------|--------|--------|
|hi4d Hug|51| ![alt](ablation_outputs/pair21_hug21_unlit/51/images_undistorted/render_v2a_7_1_4.png)| ![alt](ablation_outputs/pair21_hug21_unlit/51/images_undistorted/render_v2a_wo_comp_7_1_4.png)| ![alt](ablation_outputs/pair21_hug21_unlit/51/images_undistorted/real_4.png)
|hi4d Yoga|40|![alt](ablation_outputs/pair00_yoga00_unlit/40/images_undistorted/render_v2a_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/40/images_undistorted/render_v2a_wo_comp_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/40/images_undistorted/real_16.png)|
|hi4d Yoga|60|![alt](ablation_outputs/pair00_yoga00_unlit/60/images_undistorted/render_v2a_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/60/images_undistorted/render_v2a_wo_comp_7_1_16.png)|![alt](ablation_outputs/pair00_yoga00_unlit/60/images_undistorted/real_16.png)|





