ALLPHONEMES = ['blank', 'sil', 'PA', 'A', 'RA', 'BA', 'HA', 'MA', 'VA', 'SA', 'TA', 'U', 'O', 'NA', 'DA', 'EE', 'YA', 'TAl', 'I', 'THA', 'BHA', 'UU', 'CA', 'SHA', 'SSA', 'TTA', 'LA', 'E', 'MAl', 'GA', 'KA', 'II', 'VOCALIC_R', 'AA', 'DDA', 'NAl', 'KHA', 'DAl', 'LLA', 'NGA', 'DHA', 'AI', 'LAl', 'OO', 'JA', 'KAl', 'NNA', 'LLAl', 'TTAl', 'PAl', 'AU', 'NYA', 'CAl', 'NNAl', 'BAl', 'GAl', 'CHA', 'DDAl', 'TTHA', 'GHA', 'PHA', 'SAl', 'DDHA']

index_dict =  {'blank': 0, 'sil': 1, 'PA': 2, 'A': 3, 'RA': 4, 'BA': 5, 'HA': 6, 'MA': 7, 'VA': 8, 'SA': 9, 'TA': 10, 'U': 11, 'O': 12, 'NA': 13, 'DA': 14, 'EE': 15, 'YA': 16, 'TAl': 17, 'I': 18, 'THA': 19, 'BHA': 20, 'UU': 21, 'CA': 22, 'SHA': 23, 'SSA': 24, 'TTA': 25, 'LA': 26, 'E': 27, 'MAl': 28, 'GA': 29, 'KA': 30, 'II': 31, 'VOCALIC_R': 32, 'AA': 33, 'DDA': 34, 'NAl': 35, 'KHA': 36, 'DAl': 37, 'LLA': 38, 'NGA': 39, 'DHA': 40, 'AI': 41, 'LAl': 42, 'OO': 43, 'JA': 44, 'KAl': 45, 'NNA': 46, 'LLAl': 47, 'TTAl': 48, 'PAl': 49, 'AU': 50, 'NYA': 51, 'CAl': 52, 'NNAl': 53, 'BAl': 54, 'GAl': 55, 'CHA': 56, 'DDAl': 57, 'TTHA': 58, 'GHA': 59, 'PHA': 60, 'SAl': 61, 'DDHA': 62}

label_dict =  {0: 'blank', 1: 'sil', 2: 'PA', 3: 'A', 4: 'RA', 5: 'BA', 6: 'HA', 7: 'MA', 8: 'VA', 9: 'SA', 10: 'TA', 11: 'U', 12: 'O', 13: 'NA', 14: 'DA', 15: 'EE', 16: 'YA', 17: 'TAl', 18: 'I', 19: 'THA', 20: 'BHA', 21: 'UU', 22: 'CA', 23: 'SHA', 24: 'SSA', 25: 'TTA', 26: 'LA', 27: 'E', 28: 'MAl', 29: 'GA', 30: 'KA', 31: 'II', 32: 'VOCALIC_R', 33: 'AA', 34: 'DDA', 35: 'NAl', 36: 'KHA', 37: 'DAl', 38: 'LLA', 39: 'NGA', 40: 'DHA', 41: 'AI', 42: 'LAl', 43: 'OO', 44: 'JA', 45: 'KAl', 46: 'NNA', 47: 'LLAl', 48: 'TTAl', 49: 'PAl', 50: 'AU', 51: 'NYA', 52: 'CAl', 53: 'NNAl', 54: 'BAl', 55: 'GAl', 56: 'CHA', 57: 'DDAl', 58: 'TTHA', 59: 'GHA', 60: 'PHA', 61: 'SAl', 62: 'DDHA'}

ctc_labels = ['_', 'sil', 'PA', 'A', 'RA', 'BA', 'HA', 'MA', 'VA', 'SA', 'TA', 'U', 'O', 'NA', 'DA', 'EE', 'YA', 'TAl', 'I', 'THA', 'BHA', 'UU', 'CA', 'SHA', 'SSA', 'TTA', 'LA', 'E', 'MAl', 'GA', 'KA', 'II', 'VOCALIC_R', 'AA', 'DDA', 'NAl', 'KHA', 'DAl', 'LLA', 'NGA', 'DHA', 'AI', 'LAl', 'OO', 'JA', 'KAl', 'NNA', 'LLAl', 'TTAl', 'PAl', 'AU', 'NYA', 'CAl', 'NNAl', 'BAl', 'GAl', 'CHA', 'DDAl', 'TTHA', 'GHA', 'PHA', 'SAl', 'DDHA']

#datapath=<>
seed=25

datapath = "data"
trainSupportSet = datapath+'/train_allphn_withblank_untagged.pkl'
devSupportSet = datapath+'/dev_allphn_withblank_untagged.pkl'
trainQuerySet = datapath+'/train_xz_modfied.pkl'
devQuerySet= datapath+'/train_xz_modfied.pkl'
testQuerySet = datapath+'/test_xz_modfied.pkl'
use_cuda= True

#P way nshot
P=len(ALLPHONEMES)
Q=20
blanks_train = 620
#Kway N shot
N=20
K=len(ALLPHONEMES)
blanks_inference=620


prev_model_epochs=0#currnet model epoch number if any
prev_model_path=""#current model generated path if any

epochs=2 #Number of epochs need to be run now
model_store_path="Model_generated_"+str(Q)+"_shots_untagged"

saved_model_for_inference=model_store_path+"/model_"+str(P)+"_"+str(Q)+"_"+str(blanks_train)+"_"+str(prev_model_epochs+epochs-1)+".pth"
