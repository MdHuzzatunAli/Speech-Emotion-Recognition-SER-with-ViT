import train_ser
from train_ser import parse_arguments
import sys
import pickle
import os
import time
import random


# repeat_kfold = 2 # to  perform 10-fold for n-times with different seed
repeat_kfold = 1 # to  perform 10-fold for n-times with different seed
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

#------------PARAMETERS---------------#

# features_file = 'IEMOCAP_multi.pkl'
features_file = '/kaggle/input/iemocap/IEMOCAP_multi.pkl'

#  leave-one-speaker-out
# val_id = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F'] 
# test_id = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F'] 

# val_id = ['1M','2M','2F','3M','3F'] 
# test_id = ['1F','4M','4F','5M','5F'] 

#  leave-one-session-out
# val_id = ['1M','2M','3M','4M','5M']
# test_id = ['1F','2F','3F','4F','5F']

num_epochs  = '100'
early_stop = '8'
batch_size  = '8'
lr          = '0.00001'
random_seed = 111
gpu = '1'
gpu_ids = ['0']
# save_label = str_time#'0930_01'#'alexnet_pm_0704'

 

#Start Cross Validation
all_stat = []

cnt = 1

with open(features_file, "rb") as fin:
    features_data = pickle.load(fin)
first_9_indices = list(features_data.keys())[:9]

for repeat in range(repeat_kfold):

    random_seed +=  (repeat*100)
    seed = str(random_seed)
    
    # num_entries = 5
    
        
    

    # Randomly select 6 indices
    
        
        # if isinstance(features_data, list):  # For list-based datasets
        #     features_data = features_data[:num_entries]
        #     # print("List Data")
        # elif isinstance(features_data, dict):  # For dictionary-based datasets
        #     features_data = {k: features_data[k] for i, k in enumerate(features_data) if i < num_entries}
        #     # print("Dictionary")
        # else:
        #     raise ValueError("Unsupported data format in the .pkl file")

    # for v_id, t_id in list(zip(val_id, test_id)):
    for i in range(11):
        
        save_label = '/kaggle/working/'
        save_label = save_label + 'paper' + str(cnt)
        
        total_5_indices = random.sample(first_9_indices, 5)
        features_data_light = {key: features_data[key] for key in total_5_indices}
        v_id = total_5_indices[-1]

        train_ser.sys.argv      = [
                        
                                  'train_ser.py', 
                                #   features_data_light,
                                  '--repeat_idx', str(repeat),
                                  '--val_id',v_id, 
                                #   '--test_id', t_id,
                                  '--gpu', gpu,
                                  '--gpu_ids', gpu_ids,
                                  '--num_epochs', num_epochs,
                                  '--early_stop', early_stop,
                                  '--batch_size', batch_size,
                                  '--lr', lr,
                                  '--seed', seed,
                                  '--save_label', save_label,#,
                                  '--pretrained'
                                  ]

    
        stat = train_ser.main(parse_arguments(sys.argv[1:]), features_data_light)   
        all_stat.append(stat)    
        cnt += 1   
        # os.remove(save_label+'.pth')
    
    # with open('allstat_iemocap_'+save_label+'_'+str(repeat)+'.pkl', "wb") as fout:
    #     pickle.dump(all_stat, fout)

n_total = repeat_kfold*11
total_best_epoch, total_epoch, total_loss, total_wa, total_ua = 0, 0, 0, 0, 0

for i in range(n_total):
    # print(i, ": ", all_stat[i][0], all_stat[i][1], all_stat[i][8], all_stat[i][9], all_stat[i][10]) 
    print(i, ": ", all_stat[i][0], all_stat[i][1]) 
    total_best_epoch += all_stat[i][0]
    total_epoch += all_stat[i][1]
    # total_loss += float(all_stat[i][8])
    # total_wa += float(all_stat[i][9])
    # total_ua += float(all_stat[i][10])

# print("AVERAGE:", total_best_epoch/n_total, total_epoch/n_total, total_loss/n_total, total_wa/n_total, total_ua/n_total )
print("AVERAGE:", total_best_epoch/n_total, total_epoch/n_total)

print(all_stat)
