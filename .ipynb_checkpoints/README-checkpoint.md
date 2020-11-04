# Matching Network (MN) implementation for Continous Speech Recogniton 

In this work, we adapt th FSL paradigm *Matching Networks* (MN) to the problem of Continuous Speech Recognition.MN is a model-based FSL technique, where the embedded function is learned by prior knowledge. Matching networks carries out FSL during inference by using a small set of K samples (examples/class) to classify a test sample within a posterior estimation method based on kernel-density estimation (KDE) and k-nearest neighbor (KNN) based classification.

We adapt MN framework to continuous speech recognition (CSR) by using CTC loss function.

![Adaption of MN to CSR](images/Adaption_of_MN_to_CSR.png)

## Steps to run MN experiment for TIMIT speech corpus 

#### Prerequistes:
*   Kaldi toolkit
*   pytorch 1.0

### 1. Feature Extraction
We use Kaldi toolkit to compute features - 39 dimension MFCC (25 ms frames shifted by 10ms each time), required for training MN. 
We need to run 'timit' recipe in Kaldi as below:

cd /kaldi/egs/timit/s5/    
./run.sh  

Features :  (after applying CMVN)  
_data/train/train.scp, data/train/train.ark_   
_data/dev/dev.scp , data/dev/dev.scp_  
_data/test/test.scp , data/test/test.scp_  

Transcripts:  
_data/train/text_  
_data/dev/text_   
_data/test/text_   

### 2. Data Preparation

The data preparation folder contains 2 jupyter files - pytorch code for DNN-CTC training and code to generate support set and batch input files.

#### DNN-CTC training: 
Presently we are working with an MN formulation where in the support-set includes blank labels('_') as a class label. To generate support-set (S and S') containing frames corresponding to blank labels, we train DNN end to end using CTC loss function in pytorch. 

*    input (x)       : 39-dimension MFCC's from kaldi 
*    transcripts (z) : ground-truth corresponding to each wav file (text file from kaldi)

MN input data is of 2 types :
1. Support set (S/S') - contains frame-label pairs; Q samples for P classes.  
    format - {phonemes  : [429-dim mfcc frames] } - Phonemes include blanks ('_')  
2. Batch (B/B') - includes an utterance and the correpsonding ground-truth  
    format - {utt_id : {'feats': [39-dim MFCC] , 'labels': [ground-truth - sequence of phn indices]}}  


### 3. MN-CTC training in pytorch

We carry out to end-to-end MN training using CTC loss function. The network consists of 2 encoders 'g' and 'f' to embed the support-set samples and batch utterance.

MN consists of two encoders g and f to embed the support samples and batch utterance.
1. encoder 'g' - to embed support set frames - fed as an spectrographic patch (39x11) to a 3-layer CNN
2. encoder 'f' - uses bi-LSTM to map utterances to 256 dimension space.

t-sne plot of support-samples after embedding:

![Support set samples before and after embedding](images/t-sne_SS_plot.png)
