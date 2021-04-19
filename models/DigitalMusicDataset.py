import gzip
import json
from torch.utils.data import Dataset

class DigitalMusicDataset(Dataset):
    def __init__(self, path='dataset/Digital_Music_5.json.gz'):
        self._path = path
        self._reviews = []
        with gzip.open(path, 'rb') as g:
            for line in g:
                line = json.loads(line)
                # we only need product, reviewer ID and rating
                keys = ['asin', 'reviewerID', 'overall']
                l = {k: line[k] for k in keys}
                if len(l) == len(keys):
                    self._reviews.append(l)

    def __len__(self):
        return len(self._reviews)

    def __getitem__(self, idx):
        # Todo, based on how we define the graph
        return self._reviews[idx]
