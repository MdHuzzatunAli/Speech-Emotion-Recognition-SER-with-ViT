import train_ser
from train_ser import parse_arguments
import sys
import pickle
import os
import time

# Repeat k-fold for n-times with different seed
repeat_kfold = 2
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

#------------PARAMETERS---------------#

# Features file location
features_file = '/kaggle/input/iemocap/IEMOCAP_multi.pkl'

# Leave-one-speaker-out validation and test sets
val_id = ['1M', '2M', '2F', '3M', '3F', '4M', '4F', '5M', '5F'] 
test_id = ['1M', '2M', '2F', '3M', '3F', '4M', '4F', '5M', '5F'] 

# Hyperparameters
num_epochs = 3
early_stop = 8
batch_size = 8
lr = 0.00001
random_seed = 111
gpu = '1'
gpu_ids = ['0']
save_label = str_time  # Unique save label based on current timestamp

# Start Cross-Validation
all_stat = []

for repeat in range(repeat_kfold):
    random_seed += (repeat * 100)  # Change seed for each repeat
    seed = str(random_seed)

    # Iterate over validation and test splits
    for v_id, t_id in list(zip(val_id, test_id)):
        # Set up arguments for the training script
        train_ser.sys.argv = [
            'train_ser.py',
            features_file,
            '--repeat_idx', str(repeat),
            '--val_id', v_id,
            '--test_id', t_id,
            '--gpu', gpu,
            '--gpu_ids', gpu_ids,
            '--num_epochs', str(num_epochs),  # Ensure it's passed as a string
            '--early_stop', str(early_stop),  # Ensure it's passed as a string
            '--batch_size', str(batch_size),  # Ensure it's passed as a string
            '--lr', str(lr),  # Ensure it's passed as a string
            '--seed', seed,
            '--save_label', save_label,
            '--pretrained'
        ]

        # Train the model and collect stats
        stat = train_ser.main(parse_arguments(sys.argv[1:]))   
        all_stat.append(stat)       
        os.remove(save_label + '.pth')  # Remove model checkpoint after each fold

# Optionally save the stats if needed
# with open('allstat_iemocap_' + save_label + '_' + str(repeat) + '.pkl', "wb") as fout:
#     pickle.dump(all_stat, fout)

# Aggregate statistics across all repeats and folds
n_total = repeat_kfold * len(val_id)
total_best_epoch, total_epoch, total_loss, total_wa, total_ua = 0, 0, 0, 0, 0

# Output the stats for each fold and repeat
for i in range(n_total):
    print(i, ": ", all_stat[i][0], all_stat[i][1], all_stat[i][8], all_stat[i][9], all_stat[i][10]) 
    total_best_epoch += all_stat[i][0]
    total_epoch += all_stat[i][1]
    total_loss += float(all_stat[i][8])
    total_wa += float(all_stat[i][9])
    total_ua += float(all_stat[i][10])

# Print the averages across all repeats
print("AVERAGE:", total_best_epoch / n_total, total_epoch / n_total, total_loss / n_total, total_wa / n_total, total_ua / n_total)

print(all_stat)
