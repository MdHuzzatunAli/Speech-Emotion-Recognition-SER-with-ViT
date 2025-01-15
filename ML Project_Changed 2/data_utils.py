import pickle
import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn import preprocessing
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from collections import defaultdict

"Checking Github"
SCALER_TYPE = {'standard':'preprocessing.StandardScaler()',
               'minmax'  :'preprocessing.MinMaxScaler(feature_range=(0,1))'
              }


class TrainDataset(torch.utils.data.Dataset):
    """
    data : ndarray
        Input data of shape `N x C x H x W`, where `N` is the number of examples
        (segments), C is number of input channels (3 in the case of image), `H` is image height,
        `W` is image width
    target : ndarray
        Labels for segments (note that one utterance might contain more than
        one segments) of shape `(N,)`.
    num_classes :
        Number of classes.    
    """
    def __init__(self, data, num_classes=4):
        super(TrainDataset).__init__()
        self.data_spec = data['seg_spec']
        self.data_mfcc = data['seg_mfcc']
        self.data_audio = data['seg_audio']
        self.seg_label = data['seg_label']
        # self.target = target
        self.n_samples = len(self.seg_label)
        self.num_classes = num_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = {
            'seg_spec': self.data_spec[index], 
            'seg_mfcc': self.data_mfcc[index],
            'seg_audio': self.data_audio[index],
            'seg_label': self.seg_label[index]
            } 
        return sample
        
    def get_preds(self, preds):
        """
        Get predictions for all utterances from their segments' prediction.
        This function will accumulate the predictions for each utterance by
        taking the maximum probability along the dimension 0 of all segments
        belonging to that particular utterance.
        """     
        preds = np.argmax(preds, axis=1)
        return preds

        
    def weighted_accuracy(self, preds):
        # Check if the shapes of actual_target and preds match
        if len(self.actual_target) != len(preds):
            # Resize both arrays to the minimum length
            min_length = min(len(self.actual_target), len(preds))
            self.actual_target = self.actual_target[:min_length]
            preds = preds[:min_length]
            print(f"Resizing actual_target and preds to match the minimum length: {min_length}")
    
        # Now perform the accuracy calculation
        acc = (self.actual_target == preds).sum() / len(self.actual_target)
        return acc




    def unweighted_accuracy(self, predictions):
        """
        Calculate unweighted accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Unweighted Accuracy (UA) score.

        """


        class_acc = 0
        n_classes = 0
        for c in range(self.num_classes):
            class_pred = np.multiply(( self.seg_label == predictions),
                                     ( self.seg_label == c)).sum()
            
            if (self.seg_label == c).sum() > 0:
                 class_pred /= ( self.seg_label == c).sum()
                 n_classes += 1

                 class_acc += class_pred
            
        return class_acc / n_classes



class TestDataset(torch.utils.data.Dataset):
    """
    Holds data for a validation/test set.

    Parameters
    ----------
    data : ndarray
        Input data of shape `N x C x H x W`, where `N` is the number of examples
        (segments), C is number of input channels (3 in the case of image), `H` is image height, 
        `W` is image width
    actual_target : ndarray
        Actual target labels (labels for utterances) of shape `(U,)`, where
        `U` is the number of utterances.
    seg_target : ndarray
        Labels for segments (note that one utterance might contain more than
        one segments) of shape `(N,)`.
    num_segs : ndarray
        Array of shape `(U,)` indicating how many segments each utterance
        contains.
    num_classes :
        Number of classes.
    """
        
    def __init__(self, data, num_classes=4):
        super(TestDataset).__init__()
        # self.data = data
        self.data_spec = data['seg_spec']
        self.data_mfcc = data['seg_mfcc']
        self.data_audio = data['seg_audio']
        # self.utter_label = data['utter_label']
        # self.seg_label = data['seg_label']
        # self.seg_num = data['seg_num']
        
        self.target = data['seg_label']
        self.n_samples = len(self.target)
        self.actual_target = data['utter_label']
        self.n_actual_samples = len(self.actual_target)
        self.num_segs = data['seg_num']
        self.num_classes = num_classes


    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = {
            'seg_spec': self.data_spec[index], 
            'seg_mfcc': self.data_mfcc[index],
            'seg_audio': self.data_audio[index],
            'seg_label': self.target[index]#,
            #'utter_label': self.actual_target[index],
            #'seg_num': self.num_segs[index]
            } 
        return sample
        # return self.data[index], self.target[index]

    def get_preds(self, seg_preds):
        """
        Get predictions for all utterances from their segments' prediction.
        This function will accumulate the predictions for each utterance by
        taking the maximum probability along the dimension 0 of all segments
        belonging to that particular utterance.
        """
        preds = np.empty(
            shape=(self.n_actual_samples, self.num_classes), dtype="float")

        end = 0
        
        for v in range(self.n_actual_samples):
            start = end
            end = start + self.num_segs[v]
            
            '''
            # remove the last one for long utterances
            if self.num_segs[v] > 1:
                end = end - 1
                
            preds[v] = np.average(seg_preds[start:end], axis=0)
            
            if self.num_segs[v] > 1:
                end = end + 1
            
            
            # choose the most certain one
            tmp_seg = -1
            for seg in range(self.num_segs[v]):
                end_seg = start + seg
                if np.max(seg_preds[end_seg]) - np.min(seg_preds[end_seg]) > tmp_seg:
                    tmp_seg = np.max(seg_preds[end_seg]) - np.min(seg_preds[end_seg])
                    preds[v] = seg_preds[end_seg]
            '''  
            preds[v] = np.average(seg_preds[start:end], axis=0)
                                 
        preds = np.argmax(preds, axis=1)
        return preds


    def weighted_accuracy(self, utt_preds):
        """
        Calculate accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Accuracy score.

        """

        acc = (self.actual_target == utt_preds).sum() / self.n_actual_samples
        return acc


    def unweighted_accuracy(self, utt_preds):
        """
        Calculate unweighted accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Unweighted Accuracy (UA) score.

        """
        class_acc = 0
        n_classes = 0
        
        for c in range(self.num_classes):
            class_pred = np.multiply((self.actual_target == utt_preds),
                                     (self.actual_target == c)).sum()

        
            if (self.actual_target == c).sum() > 0:    
                class_pred /= (self.actual_target == c).sum()
                n_classes += 1
                class_acc += class_pred
        
        return class_acc / n_classes

    
    def confusion_matrix_iemocap(self, utt_preds):
        """Compute confusion matrix given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        """
        conf = confusion_matrix(self.actual_target, utt_preds)
        
        # Make confusion matrix into data frame for readability
        conf_fmt = pd.DataFrame({"ang": conf[:, 0], "sad": conf[:, 1],
                             "hap": conf[:, 2], "neu": conf[:, 3]})
        conf_fmt = conf_fmt.to_string(index=False)
        print(conf_fmt)
        return (conf, conf_fmt)


from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class SERDataset(Dataset):
    def __init__(self, features_data, num_classes=4,
                 val_speaker_id='1M', test_speaker_id='1F', 
                 oversample=False, transform=None):
        
        self.num_classes = num_classes
        self.transform = transform

        # Initialize data dictionaries
        self.train_data = defaultdict()
        self.val_data = defaultdict()
        self.test_data = defaultdict()

        # Extract and store training data in dictionaries
        for speaker_id in features_data.keys():
            if speaker_id in [val_speaker_id, test_speaker_id]:
                continue
            # Store data in train_data
            if self.train_data.get("seg_spec") is None:
                self.train_data["seg_spec"] = features_data[speaker_id]['seg_spec'].astype(np.float32)
                self.train_data["seg_mfcc"] = features_data[speaker_id]['seg_mfcc'].astype(np.float32)
                self.train_data["seg_audio"] = features_data[speaker_id]['seg_audio'].astype(np.float32)
                self.train_data["seg_label"] = features_data[speaker_id]['seg_label'].astype(np.int64)
                self.train_data["utter_label"] = features_data[speaker_id]['utter_label'].astype(np.int64)
                self.train_data["seg_num"] = np.array([features_data[speaker_id]['seg_spec'].shape[0]])  # Initialize as an array
            else:
                self.train_data["seg_spec"] = np.concatenate((self.train_data["seg_spec"],
                                                              features_data[speaker_id]['seg_spec'].astype(np.float32)), axis=0)
                self.train_data["seg_mfcc"] = np.concatenate((self.train_data["seg_mfcc"],
                                                              features_data[speaker_id]['seg_mfcc'].astype(np.float32)), axis=0)
                self.train_data["seg_audio"] = np.concatenate((self.train_data["seg_audio"],
                                                              features_data[speaker_id]['seg_audio'].astype(np.float32)), axis=0)
                self.train_data["seg_label"] = np.concatenate((self.train_data["seg_label"],
                                                              features_data[speaker_id]['seg_label'].astype(np.int64)), axis=0)
                self.train_data["utter_label"] = np.concatenate((self.train_data["utter_label"],
                                                                 features_data[speaker_id]['utter_label'].astype(np.int64)), axis=0)
                self.train_data["seg_num"] = np.concatenate((self.train_data["seg_num"],
                                                              np.array([features_data[speaker_id]['seg_spec'].shape[0]])), axis=0)  # Initialize as array and concatenate

        # Similarly, add 'seg_num' for val_data and test_data
        self.val_data["seg_spec"] = features_data[val_speaker_id]['seg_spec'].astype(np.float32)
        self.val_data["seg_mfcc"] = features_data[val_speaker_id]['seg_mfcc'].astype(np.float32)
        self.val_data["seg_audio"] = features_data[val_speaker_id]['seg_audio'].astype(np.float32)
        self.val_data["seg_label"] = features_data[val_speaker_id]['seg_label'].astype(np.int64)
        self.val_data["utter_label"] = features_data[val_speaker_id]['utter_label'].astype(np.int64)
        self.val_data["seg_num"] = np.array([features_data[val_speaker_id]['seg_spec'].shape[0]])  # Initialize as array

        self.test_data["seg_spec"] = features_data[test_speaker_id]['seg_spec'].astype(np.float32)
        self.test_data["seg_mfcc"] = features_data[test_speaker_id]['seg_mfcc'].astype(np.float32)
        self.test_data["seg_audio"] = features_data[test_speaker_id]['seg_audio'].astype(np.float32)
        self.test_data["seg_label"] = features_data[test_speaker_id]['seg_label'].astype(np.int64)
        self.test_data["utter_label"] = features_data[test_speaker_id]['utter_label'].astype(np.int64)
        self.test_data["seg_num"] = np.array([features_data[test_speaker_id]['seg_spec'].shape[0]])  # Initialize as array

        # Normalize dataset
        self._normalize('minmax')

        if oversample:
            print('\nPerform training dataset oversampling')
            datar, labelr = random_oversample(self.train_data["seg_spec"], self.train_data["utter_label"])
            self.train_data["seg_spec"] = datar
            self.train_data["utter_label"] = labelr

        # Convert to 3-channel images (RGB)
        self.train_data["seg_spec"] = self._spec_to_gray(self.train_data["seg_spec"])
        self.val_data["seg_spec"] = self._spec_to_gray(self.val_data["seg_spec"])
        self.test_data["seg_spec"] = self._spec_to_gray(self.test_data["seg_spec"])

        # Apply transforms if provided
        if self.transform:
            self.train_data["seg_spec"] = [self.transform(x) for x in self.train_data["seg_spec"]]
            self.val_data["seg_spec"] = [self.transform(x) for x in self.val_data["seg_spec"]]
            self.test_data["seg_spec"] = [self.transform(x) for x in self.test_data["seg_spec"]]

    def __len__(self):
        return len(self.train_data["seg_spec"])

    def __getitem__(self, idx):
        sample = {
            'spec': self.train_data["seg_spec"][idx],
            'label': self.train_data["seg_label"][idx]
        }
        return sample

    def _normalize(self, scaling):
        self.train_data["seg_spec"] = self._apply_scaling(self.train_data["seg_spec"], scaling)
        self.val_data["seg_spec"] = self._apply_scaling(self.val_data["seg_spec"], scaling)
        self.test_data["seg_spec"] = self._apply_scaling(self.test_data["seg_spec"], scaling)
        print(f'Dataset normalized with {scaling} scaler')
    
    def _apply_scaling(self, data, scaling):
        # Apply normalization (e.g., min-max scaling or standard scaling)
        return data  # Placeholder

    def _spec_to_gray(self, data):
        # Convert spectrogram to grayscale
        return data  # Placeholder

    def get_train_dataset(self):
        return TrainDataset(self.train_data, num_classes=self.num_classes)
    
    def get_val_dataset(self):
        return TestDataset(self.val_data, num_classes=self.num_classes)
    
    def get_test_dataset(self):
        return TestDataset(self.test_data, num_classes=self.num_classes)
 

def random_oversample(data, labels):
    print('\tOversampling method: Random Oversampling')
    ros = RandomOverSampler(random_state=0,sampling_strategy='minority')

    n_samples = data.shape[0]
    fh = data.shape[2]
    fw = data.shape[3]
    n_features= fh*fw
        
    data = np.squeeze(data,axis=1)
    data = np.reshape(data,(n_samples, n_features))
    data_resampled, label_resampled = ros.fit_resample(data, labels)
    n_samples = data_resampled.shape[0]
    data_resampled = np.reshape(data_resampled,(n_samples,fh,fw))
    data_resampled = np.expand_dims(data_resampled, axis=1)
    
    return data_resampled, label_resampled


