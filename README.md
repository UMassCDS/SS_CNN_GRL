# Prediction of Off-Fault Deformation from Experimental Strike-slip Fault Structures using the Convolutional Neural Networks 

Where the earth deforms at the boundaries between tectonic plates, some of the deformation is taken up as localized slip along fault surfaces and some of the deformation is distributed around the fault. This distributed deformation is very hard to measure in the Earth’s crust. To get around this challenge, we create faults in the laboratory and use the direct measurements of the distributed deformation off of faults to train a machine learning model. The trained model performs well at predicting distributed off-fault deformation from the fault geometry.

[![DOI](https://zenodo.org/badge/381946830.svg)](https://zenodo.org/badge/latestdoi/381946830)

## Introduction 


This repository contains the end-to-end codes to predict off fault deformation directly from fault maps using CNN. This repository contains:

- ```SSF_CNN_GRL/Data_Processing.ipynb```: code to convert raw .mat data into appproipriated size, generate labeled input dataset, and split into 3 subsets.

- ```SSF_CNN_GRL/Model/Train.ipynb```code to setup a training session for CNN. This code define model architecture and utilize KerasDataGenerator and KerasImageAugmentation to feed appropriated augmented data for trainining.

- ```SSF_CNN_GRL/Model/Evaluate.ipynb```: code to apply the 'best_model' on unseen data (Evaluation Dataset and Test Dataset) to predict KE, as well as reaffirm its ability to accurately predict Train Dataset.  



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
git clone https://github.com/laainam/SS_CNN_GRL.git
cd SS_CNN_GRL
```
- **Raw Matlab** experiment files can be download as [raw_matlab.zip](https://figshare.com/s/3ea3c27706a7aab3d01c). It should be unzipped into 'SSF_CNN_GRL/raw_data/raw_matlab' folder.


## Run
- Run ```SSF_CNN_GRL/Data_Processing.ipynb``` to process raw matlab into ready-to-use .npy input files (labeled).  
	* Cropped .npy files with label embeded in file names saved in 'SSF_CNN_GRL/processed_input_data/slice_npy' folder
	* Split dataset can be called using 'train_master.txt', 'eval_master.txt', 'test_master.txt' located in 'SSF_CNN_GRL/processed_input_data/split_master' folder
- Run ```SSF_CNN_GRL/Model/Train``` 
 	* SSF_CNN_GRL/Model/experimements/archive_final_run store the post-trained models for reference.
		* Do not re-train or save over, to be used for evaluation.  
	* Each training session require a set of hyperparameters, which are stored in 'params.json'. 
 		* 'SSF_CNN_GRL/Model/experimements/run1 contains a 'params.json' file ready for training. 
      *  With random initialization, model performance may differ from the archive's performance, but should maintain consistent performance using similar optimal hyperparameter sets. 
      *  Manually update selectedE, using 'Epoch' that show higest 'Eval_2SD_Accuracy'
- Run ```SSF_CNN_GRL/Model/Eval```
	* To predict unseen dataset using post-trained 'best model'
  
## Customized Loss Function  & Accuracy Metric
Convolutional Neural Networks (CNNs) trained using experimental strike-slip fault maps can provide a useful way to describe the complex and non-linear relationship between active fault trace complexity and kinematic efficiency. Learning directly from fault maps eliminates the need to prescribe exact equations to describe complex failure behaviors. The proposed CNNs learn how active fault traces relate to KE by minimizing a custom loss function L based on a normalized mean square error as shown in Eq. 1

<img src="https://github.com/laainam/SS_CNN_GRL/blob/main/image/eq1.png" width="500">
The mean square error (MSE) is the squared difference of the estimated values (KE prediction, yi) and the truth (KE label, yi). A small value of ensures a non-zero divisor. Our custom loss function scales MSE with the squared standard deviation of KE (SD), allowing the model to learn more precisely where we have the most confidence while relaxing the learning conditions where uncertainties are high

We assess the performance of our CNN networks by considering the prediction as correct if the absolute difference of the predicted KE and the true KE label fit within two standard deviations of the label (Eq. 2). 
<img src="https://github.com/laainam/SS_CNN_GRL/blob/main/image/eq2.png" width="300">

## CNN Architecture and Training Performance

To ensure that the trained CNN can generalize to unseen data, we use the minimum loss (Eq. 1) of the evaluation dataset to guide tuning of the hyperparameters. The best model, and all repeated training runs illustrate a good fit, and the CNN model stops improving after approximately 50 training epochs, where we impose an early stopping of the training process(Fig. 2b). Additionally, we confirm the repeatability of the models’ performance by reproducing mini-batch accuracy over 90% on all training using the same set of hyperparameters (Table S2) while varying the randomized initialization. 

<img src="https://github.com/laainam/SS_CNN_GRL/blob/main/image/fig2.png" width="800">

## Prediction Performance 

Applying the selected CNN’s model for prediction tasks, we reach high performance of 96.7% and 96.1% accuracy (Eq.2) in training and evaluation datasets respectively. Similarly,  prediction on an unseen test dataset yields satisfactory performance of 90.9% accuracy. These correct predictions for the majority of the dataset extensively represent experiments with the full range of applied loading rates, basal boundary conditions and stages of fault evolution. On the other hand, the clusters of outliers from more matured faults seem to correlate to individual experiments within a specific KE range. 

<img src="https://github.com/laainam/SS_CNN_GRL/blob/main/image/fig3.png" width="800">

## Summary
While seismic hazard analyses benefit from estimates of off-fault deformation, we do not have reliable ways to measure the portion of strain that is accommodated off faults. Here, we offer an alternative approach for KE prediction using a 2D Convolutional Neural Network, that is trained directly on images of fault maps produced by fault experiments scaled to simulate crustal strike-slip faults. Our dataset captures the whole evolution of strike-slip faults and allows precise calculation of off-fault deformation (1-KE).  We use a custom loss function and custom accuracy, which fully utilize both the KE labels and their standard deviation. We tune the set of hyperparameters to optimize our CNN training. The final CNN model has the ability to predict on an unseen test dataset with 91% accuracy. Lastly, the match of the CNN to crustal fault maps with off-fault deformation estimates shows the potential for applying experimentally trained CNNs to crustal faults. 

## Co-authors / Full Paper 
- L. Chaipornkaew, H. Elston, M. Cooke, T. Mukerji, S. Graham 
- The full extent of this work can be found here (link to GRL paper).



