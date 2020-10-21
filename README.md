# Matching Network (MN) implementation for Continous Speech Recogniton 

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
data/train/train.scp, data/train/train.ark
data/dev/dev.scp , data/dev/dev.scp
data/test/test.scp , data/test/test.scp

Transcripts:
data/train/text
data/dev/text
data/test/text

### 2. Data Preparation

The data preparation folder contains 2 jupyter files - pytorch code for DNN-CTC training and code to generate support set and batch input files.

#### DNN-CTC training: 
Presently we are working with an MN formulation where in the support-set includes blank labels('_') as a class label. To generate support-set (S and S') containing frames corresponding to blank labels, we train DNN end to end using CTC loss function in pytorch. 

*    input (x)       : 39-dimension MFCC's from kaldi 
*    transcripts (z) : ground-truth corresponding to each wav file (text file from kaldi)

MN input data is of 2 types :
1. Support set (S/S') - contains frame-label pairs; Q samples for P classes.
    *  format - {phonemes  : [429-dim mfcc frames] } - Phonemes include blanks ('_')
2. Batch (B/B') - includes an utterance and the correpsonding ground-truth
    *  format - {utt_id : {'feats': [39-dim MFCC] , 'labels': [ground-truth - sequence of phn indices]}}


