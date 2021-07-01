# SSF-CNN

## Introduction 
This repository contains: 
- **Data_Processing** codes generate appropriated sized data and calculate label for training directly from raw matlab image in a format ready for DataGenerator to feed mini-batch into training.  

- **data_display** codes generate visualization to cross check that slicing window and label are correctly processed. 
- **train_KE** codes utilize KerasDataGenerator and KerasImageAugmentation to train the model, in which its weights are adjusted solely based on TrainingData. However, to ensure that the model is not overtrained, EvalData is used to select 'best_model'
- **evaluate_KE** codes illustrate how well the 'best_model' performed on unseen data (EvalData and TestData) as well as reaffirm that it can make accurate prediction in TrainData. This code has ability to display outliers and allow users to understand why selected dataset have a larger errors (Difference between prediction and label) than another. 


## Background
Crustal deformation occurs both as localized slip along faults and distributed deformation between active faults via a range of processes including folding, development of pervasive cleavage/foliation and/or slip along fractures within fault damage. Estimates of coseismic off fault deformation along strike-slip faults confirm the supposition that faults with smoother traces can more efficiently accommodate strike slip than faults with rough/complex traces. This hypothesis is also supported by scaled physical experiments of strike-slip fault evolution that directly document that as faults mature from echelon segments to smoother through-going faults, the % of fault slip quantified as kinematic efficiency (1- % off fault deformation) increases. 

In this study, we propose to harness machine learning on rich experimental time series data to provide estimates of kinematic efficiency directly from pattern of active strike-slip fault trace. Physical experiments that are scaled to simulate crustal strike-slip fault development allow direct and detailed observation of both active fault trace and kinematic efficiency under a range of conditions.

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
- **Raw Matlab** experiment files can be download as [raw_matlab.zip](https://drive.google.com/file/d/1qWxvNuwICb0-evqktYHDmKgWw5Hf-URi/view?usp=sharing). It should be unzipped into 'raw_data/raw_matlab' folder inside the SSF-CNN project.

## Run
Properly set up raw_data folder in SSF-CNN in the right higherachy. Here are step-by-steps from raw data to prediction as follows
- ...

## Image Processing

![Image of raw experimental channels](https://github.com/laainam/SSF-CNN/blob/master/image/raw_exp_img.png)

