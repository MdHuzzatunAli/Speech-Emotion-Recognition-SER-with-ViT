class SERDataset(Dataset):
    """
    Wrapper for both `TrainDataset` and `TestDataset`, which loads and pre-process
    speech spectrograms into `Dataset` objects.

    This also assigns the dataset into train, validation, test dataset based on IEMOCAP cross-validation
    arrangement. There are 10 speakers in total (5 sessions x  2 speakers per session) and the IDs assigned
    are 1F, 1M, 2F, 2M, 3F, 3M, 4F, 4M, 5F, 5M.

    Parameters
    ----------
    features_data
        Spectrograms extracted using `extract_features.py`, labels
    num_classes
        Number of emotion classes
    val_speaker_id
        ID of speaker to be used as validation in kfold cross-validation
    test_speaker_id
        ID of speaker to be used as test in kfold cross-validation 
    oversample : bool
        Set 'True' to apply random dataset oversampling to balance the classes
    transform
        Optional transform to be applied on a sample
    """
    def __init__(self, features_data, num_classes=4,
                 val_speaker_id='1M', test_speaker_id='1F', 
                 oversample=False, transform=None):
        
        self.num_classes = num_classes  # Ensure that num_classes is assigned here
        self.transform = transform

        # Extracting and processing training data
        self.train_spec_data, self.train_mfcc_data, self.train_audio_data, self.train_seg_labels, self.train_labels = None, None, None, None, None
        for speaker_id in features_data.keys():
            if speaker_id in [val_speaker_id, test_speaker_id]:
                continue
            if self.train_mfcc_data is None:
                self.train_spec_data = features_data[speaker_id]['seg_spec'].astype(np.float32)
                self.train_mfcc_data = features_data[speaker_id]['seg_mfcc'].astype(np.float32)
                self.train_audio_data = features_data[speaker_id]['seg_audio'].astype(np.float32)
            else:
                self.train_spec_data = np.concatenate((self.train_spec_data, 
                                            features_data[speaker_id]['seg_spec'].astype(np.float32)), axis=0)
                self.train_mfcc_data = np.concatenate((self.train_mfcc_data, 
                                            features_data[speaker_id]['seg_mfcc'].astype(np.float32)), axis=0)
                self.train_audio_data = np.concatenate((self.train_audio_data, 
                                            features_data[speaker_id]['seg_audio'].astype(np.float32)), axis=0)

            if self.train_seg_labels is None:
                self.train_seg_labels = features_data[speaker_id]['seg_label'].astype(np.int64)
                self.train_labels = features_data[speaker_id]['utter_label'].astype(np.int64)
            else:
                self.train_seg_labels = np.concatenate((self.train_seg_labels,
                                               features_data[speaker_id]['seg_label'].astype(np.int64)), axis=0)
                self.train_labels = np.concatenate((self.train_labels,
                                               features_data[speaker_id]['utter_label'].astype(np.int64)), axis=0)
        
        # Store the training data
        self.train_data = defaultdict()
        self.train_data["seg_spec"] = self.train_spec_data
        self.train_data["seg_mfcc"] = self.train_mfcc_data
        self.train_data["seg_audio"] = self.train_audio_data
        self.train_data["seg_label"] = self.train_seg_labels

        # Validation data (similar to training data)
        self.val_data = defaultdict()  # Create val_data dictionary
        self.val_data["seg_spec"] = features_data[val_speaker_id]['seg_spec'].astype(np.float32)
        self.val_data["seg_mfcc"] = features_data[val_speaker_id]['seg_mfcc'].astype(np.float32)
        self.val_data["seg_audio"] = features_data[val_speaker_id]['seg_audio'].astype(np.float32)
        self.val_data["seg_label"] = features_data[val_speaker_id]['seg_label'].astype(np.int64)

        # Test data (similar to training data)
        self.test_data = defaultdict()  # Create test_data dictionary
        self.test_data["seg_spec"] = features_data[test_speaker_id]['seg_spec'].astype(np.float32)
        self.test_data["seg_mfcc"] = features_data[test_speaker_id]['seg_mfcc'].astype(np.float32)
        self.test_data["seg_audio"] = features_data[test_speaker_id]['seg_audio'].astype(np.float32)
        self.test_data["seg_label"] = features_data[test_speaker_id]['seg_label'].astype(np.int64)

        # Normalize dataset (example, could be min-max scaling or standard scaling)
        self._normalize('minmax')

        if oversample:
            print('\nPerform training dataset oversampling')
            datar, labelr = random_oversample(self.train_spec_data, self.train_labels)
            self.train_spec_data = datar
            self.train_labels = labelr

        # Convert to 3-channel images (RGB)
        self.train_spec_data = self._spec_to_gray(self.train_spec_data)
        self.val_spec_data = self._spec_to_gray(self.val_spec_data)
        self.test_spec_data = self._spec_to_gray(self.test_spec_data)

        # Apply transforms if provided
        if self.transform:
            self.train_spec_data = [self.transform(x) for x in self.train_spec_data]
            self.val_spec_data = [self.transform(x) for x in self.val_spec_data]
            self.test_spec_data = [self.transform(x) for x in self.test_spec_data]

    def __len__(self):
        return len(self.train_spec_data)

    def __getitem__(self, idx):
        sample = {
            'spec': self.train_spec_data[idx],
            'label': self.train_seg_labels[idx]
        }
        return sample
    
    def _normalize(self, scaling):
        input_range = self._get_data_range()
        self.train_spec_data = self._apply_scaling(self.train_spec_data, scaling)
        self.val_spec_data = self._apply_scaling(self.val_spec_data, scaling)
        self.test_spec_data = self._apply_scaling(self.test_spec_data, scaling)
        print(f'Dataset normalized with {scaling} scaler')
    
    def _apply_scaling(self, data, scaling):
        # Normalization logic goes here (e.g., min-max scaling or standard scaling)
        return data  # Placeholder, apply scaling here
    
    def _get_data_range(self):
        # Calculate min and max values for normalization
        return [0, 1]  # Placeholder range
    
    def _spec_to_gray(self, data):
        # Convert spectrogram to grayscale
        return data  # Placeholder method for converting to gray images

    def get_train_dataset(self):
        return TrainDataset(self.train_data, num_classes=self.num_classes)
    
    def get_val_dataset(self):
        return TestDataset(self.val_data, num_classes=self.num_classes)  # Now using self.val_data
    
    def get_test_dataset(self):
        return TestDataset(self.test_data, num_classes=self.num_classes)
