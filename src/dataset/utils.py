"""
Utility functions for data preparation.
"""
import numpy as np
from typing import Union
from datetime import datetime
from pytz import timezone
import os
from typing import Any


class Index:
    def __init__(self, indices:Union[int,list], batch_size: int,
                 shuffle: bool=True) -> None:
        if type(indices)==int:
            indices = np.arange(indices)
        self.indices = indices
        self.num_samples = len(indices)
        self.batch_size = batch_size
        self.pointer = 0
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle

    def get_batch_ind(self):
        """Get indices for next batch."""
        start, end = self.pointer, min(self.num_samples, self.pointer + self.batch_size)
        # If we have a full batch within this epoch, then get it.
        if end==self.num_samples:
            self.pointer = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        else:
            self.pointer = end
        return self.indices[start:end]

class CycleIndex:
    """Class to generate batches of training ids, 
    shuffled after each epoch.""" 
    def __init__(self, indices:Union[int,list], batch_size: int,
                 shuffle: bool=True) -> None:
        if type(indices)==int:
            indices = np.arange(indices)
        self.indices = indices
        self.num_samples = len(indices)
        self.batch_size = batch_size
        self.pointer = 0
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle

    def get_batch_ind(self):
        """Get indices for next batch."""
        start, end = self.pointer, self.pointer + self.batch_size
        # If we have a full batch within this epoch, then get it.
        if end <= self.num_samples:
            if end==self.num_samples:
                self.pointer = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
            else:
                self.pointer = end
            return self.indices[start:end]
        # Otherwise, fill the batch with samples from next epoch.
        last_batch_indices_incomplete = self.indices[start:]
        remaining = self.batch_size - (self.num_samples-start)
        self.pointer = remaining
        if self.shuffle:
            np.random.shuffle(self.indices)
        return np.concatenate((last_batch_indices_incomplete, 
                               self.indices[:remaining]))

def get_curr_time() -> str:
    """Get current date and time in Shanghai timezone as str."""
    return datetime.now().astimezone(
            timezone('Asia/Shanghai')).strftime("%d/%m/%Y %H:%M:%S")

class Logger: 
    """Class to write message to both output_dir/filename.txt and terminal."""
    def __init__(self, output_dir: str=None, filename: str=None) -> None:
        if filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message: Any, show_time: bool=True) -> None:
        "write the message"
        # message = str(message)
        if show_time:
            # if message starts with \n, print the \n first before printing time
            if message.startswith('\n'): 
                message = '\n'+get_curr_time()+' >> '+message[1:]
            else:
                message = get_curr_time()+' >> '+message
        print(message)
        if hasattr(self, 'log'):
            with open(self.log, 'a') as f:
                f.write(message+'\n')
                f.flush()  # 立即刷新到磁盘