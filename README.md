## Introduction 
This repository contains: 
- **SSF_CNN_GRL/Data_Processing** codes generate labeled input dataset and split into 3 subsets.
    * Convert and crop raw .mat into appropriately-sized .npy input files
    * Calculate KE label for each input
    * Split data into train:eval:test based on specified criteria 

- **SSF_CNN_GRL/Model/Train** codes utilize KerasDataGenerator and KerasImageAugmentation to train the model, in which its weights are adjusted solely based on Training Dataset. However, we select 'best_model' based on the performance of the Evaluation dataset. 

- **SSF_CNN_GRL/Model/Evaluate**  codes apply the 'best_model' on unseen data (Evaluation Dataset and Test Dataset) to predict KE, as well as reaffirm its ability to accurately  predict Train Dataset.  

## Background
Crustal deformation occurs both as localized slip along faults and distributed deformation between active faults via a range of processes including folding, development of pervasive cleavage/foliation and/or slip along fractures within fault damage. Estimates of coseismic off fault deformation along strike-slip faults confirm the supposition that faults with smoother traces can more efficiently accommodate strike slip than faults with rough/complex traces. This hypothesis is also supported by scaled physical experiments of strike-slip fault evolution that directly document that as faults mature from echelon segments to smoother through-going faults, the % of fault slip quantified as kinematic efficiency (1- % off fault deformation) increases. 

In this study, we propose to harness machine learning on rich experimental time series data to provide estimates of kinematic efficiency directly from pattern of active strike-slip fault trace. Physical experiments that are scaled to simulate crustal strike-slip fault development allow direct and detailed observation of both active fault trace and kinematic efficiency under a range of condition
<img src="https://github.com/laainam/SS_CNN_GRL/blob/main/image/fig1.png" width="800">


## Environment Requirement 
```ruby
pip install -r requirement.txt
```

## Download
- **Code** Clone from github
```ruby
git clone https://github.com/laainam/SSF-CNN.git
cd SSF-CNN
```
- **Raw Matlab** experiment files can be download as [raw_matlab.zip](https://figshare.com/s/3ea3c27706a7aab3d01c). It should be unzipped into 'SSF_CNN_GRL/raw_data/raw_matlab' folder.

## Run
- Run **SSF_CNN_GRL/Data_Processing.ipynb** to process raw matlab into ready-to-use .npy input files (labeled).  
	* Cropped .npy files with label embeded in file names saved in 'SSF_CNN_GRL/processed_input_data/slice_npy' folder
	* Split dataset can be called using 'train_master.txt', 'eval_master.txt', 'test_master.txt' located in 'SSF_CNN_GRL/processed_input_data/split_master' folder
- Run **SSF_CNN_GRL/Model/Train**. 
 	* SSF_CNN_GRL/Model/experimements/archive_final_run store the post-trained models for reference.
		* Do not re-train or save over, to be used for evaluation.  
	* Each training session require a set of hyperparameters, which are stored in 'params.json'. 
 		* 'SSF_CNN_GRL/Model/experimements/run1 contains a 'params.json' file ready for training. 
      *  With random initialization, model performance may differ from the archive's performance, but should maintain consistent performance using similar optimal hyperparameter sets. 
      *  Manually update selectedE, using 'Epoch' that show higest 'Eval_2SD_Accuracy'
- Run **SSF_CNN_GRL/Model/Eval**.
	* To predict unseen dataset using post-trained 'best model'
  
## CNN Architecture and Training Performance 

<img src="https://github.com/laainam/SS_CNN_GRL/blob/main/image/fig2.png" width="800">

## Prediction Performance 
<img src="https://github.com/laainam/SS_CNN_GRL/blob/main/image/fig3.png" width="800">

## Summary
While seismic hazard analyses benefit from estimates of off-fault deformation, we do not have reliable ways to measure the portion of strain that is accommodated off faults. Here, we offer an alternative approach for KE prediction using a 2D Convolutional Neural Network, that is trained directly on images of fault maps produced by fault experiments scaled to simulate crustal strike-slip faults. Our dataset captures the whole evolution of strike-slip faults and allows precise calculation of off-fault deformation (1-KE).  We use a custom loss function and custom accuracy, which fully utilize both the KE labels and their standard deviation. We tune the set of hyperparameters to optimize our CNN training. The final CNN model has the ability to predict on an unseen test dataset with 91% accuracy. Lastly, the match of the CNN to crustal fault maps with off-fault deformation estimates shows the potential for applying experimentally trained CNNs to crustal faults. 

## Co-authors / Full Paper 
- L. Chaipornkaew, H. Elston, M. Cooke, T. Mukerji, S. Graham 
- The full extent of this work can be found here (link to GRL paper).



