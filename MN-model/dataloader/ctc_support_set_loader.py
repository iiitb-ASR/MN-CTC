
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np
import random



class stackup(object):
        
    def __call__(self, support_set):
        XS,yS,ySlabels = [],[],[]
        
        for i, details in enumerate(support_set):
            for it in details:
                XS.append(np.array(it[2]).reshape(11,39))
                yS.append(it[1])
                ySlabels.append(it[0])
       
        return XS,yS,ySlabels

class SupportDataSet(Dataset):
    def __init__(self, supportdatafile,kway, nshot, bshot, nqueries, phonemes, index_dict, label_dict, transform=None): 
        with open(supportdatafile,'rb') as f1:
            self.data=pickle.load(f1)
            
        self.phonemes=phonemes
        self.index_dict = index_dict
        self.label_dict=label_dict
        self.kway=kway
        self.nshot=nshot
        self.bshot=bshot
        self.nqueries = nqueries
        
        self.transform = transform

    def __len__(self):
        length=0
        for phn in self.phonemes:
            length += len(self.data[phn])
        return length
        

    def __getitem__(self, ix):
        S_set = []
        for classno in range (len(self.phonemes)):
            new_details=[]
            eachphone = self.label_dict[classno]  

            if eachphone == 'blank':  
                blank_details=self.data[eachphone]
                if len(blank_details) < self.bshot:
                    raise Exception("Data Error in T set:no of blank samples less than",self.bshot) #bshot : no:of blank samples
                tmp=random.sample(blank_details,self.bshot)
            else:
                details=self.data[eachphone]
                tmp = []
                if(len(details)>=self.nshot):
                    tmp=random.sample(details,self.nshot)
                else:
                    while((len(tmp)+len(details))<self.nshot):
                        tmp.extend(details)

                    tmp.extend(random.sample(details,self.nshot-len(tmp)))

            for item in tmp:
                new_details.append([eachphone, self.index_dict[eachphone], item])
            S_set.append(new_details)
        
        if self.transform:
            S_set = self.transform(S_set)

        return S_set
    
def collate_wrapper(batch):
    X_set=[]
    y_set=[]
    ylabels_set=[]
    
    for i, item in enumerate(batch):
        X,y,ylabels=item
        X_set.append(X)
        y_set.append(y)
        ylabels_set.append(ylabels)
    
    XX=np.asarray(X_set, dtype=np.float32)
    X=torch.from_numpy(XX)
    X=torch.unsqueeze(X,2)
        
    y=torch.from_numpy(np.asarray(y_set, dtype=np.long))
    ylabels=ylabels_set
    batch=[X,y,ylabels]
    
    return batch