ALLPHONEMES = ['blank','sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']

index_dict = {'blank': 0, 'sil': 1, 'aa': 2, 'ae': 3, 'ah': 4, 'aw': 5, 'ay': 6, 'b': 7, 'ch': 8, 'd': 9, 'dh': 10, 'dx': 11, 'eh': 12, 'er': 13, 'ey': 14, 'f': 15, 'g': 16, 'hh': 17, 'ih': 18, 'iy': 19, 'jh': 20, 'k': 21, 'l': 22, 'm': 23, 'n': 24, 'ng': 25, 'ow': 26, 'oy': 27, 'p': 28, 'r': 29, 's': 30, 'sh': 31, 't': 32, 'th': 33, 'uh': 34, 'uw': 35, 'v': 36, 'w': 37, 'y': 38, 'z': 39}

label_dict = {0: 'blank', 1: 'sil', 2: 'aa', 3: 'ae', 4: 'ah', 5: 'aw', 6: 'ay', 7: 'b', 8: 'ch', 9: 'd', 10: 'dh', 11: 'dx', 12: 'eh', 13: 'er', 14: 'ey', 15: 'f', 16: 'g', 17: 'hh', 18: 'ih', 19: 'iy', 20: 'jh', 21: 'k', 22: 'l', 23: 'm', 24: 'n', 25: 'ng', 26: 'ow', 27: 'oy', 28: 'p', 29: 'r', 30: 's', 31: 'sh', 32: 't', 33: 'th', 34: 'uh', 35: 'uw', 36: 'v', 37: 'w', 38: 'y', 39: 'z'}

ctc_labels = ['_', 'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']

#datapath=<>
seed=25

datapath = "data"
trainSupportSet = datapath+'/train_SS_dim429.pkl'
devSupportSet = datapath+'/dev_SS_dim429.pkl'
trainQuerySet = datapath+'/train_xz.pkl'
devQuerySet= datapath+'/train_xz.pkl'
testQuerySet = datapath+'/test_xz.pkl'
use_cuda= True

#P way nshot
P=len(ALLPHONEMES)
Q=20
blanks_train = (P-1)*int(Q/2)
#Kway N shot
N=20
K=len(ALLPHONEMES)
blanks_inference=(K-1)*int(N/2)


prev_model_epochs=0#currnet model epoch number if any
prev_model_path=""#current model generated path if any

epochs=2 #Number of epochs need to be run now
model_store_path="Model_generated_"+str(Q)+"_shots_untagged"

saved_model_for_inference=model_store_path+"/model_"+str(P)+"_"+str(Q)+"_"+str(blanks_train)+"_"+str(prev_model_epochs+epochs-1)+".pth"
