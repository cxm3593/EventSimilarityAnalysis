import numpy as np
import pandas as pd
import h5py

class EventDataManager:

    def __init__(self):
        self.data_counter = 0
        self.event_data_dict:dict = {}

        self.opened_files:list = []

    def load_event_data_h5(self, file_path, dataset_name="events", data_key=None):
        '''
        Loads the event data from the h5 file and returns the dataset.
        
        Args:
            file_path: The path to the h5 file.
            dataset_name: The name of the dataset to load.
            data_key: The key to use for the dataset. If None, a new key will be generated.
        '''

        file = h5py.File(file_path, 'r')
        self.opened_files.append(file)

        dataset = file[dataset_name]

        if data_key is None:
            data_key = self.data_counter
            self.data_counter += 1

        self.event_data_dict[data_key] = dataset

        return dataset

    def select_events_by_time_window(self, dataset, start_time, end_time):

        events_selected_subset = dataset[(dataset['t'] >= start_time) & (dataset['t'] <= end_time)]
        return events_selected_subset

    

    