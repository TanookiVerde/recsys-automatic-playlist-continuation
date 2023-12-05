
from json import dump, load

class IncrementalEncoder():

    def __init__(self):
        self.encoding = {}
        self.labels = []
        self.last_index = 0
    
    def fit_one(self, value):
        if value in self.encoding.keys():
            return self.encoding[value]
        else:
            self.encoding[value] = self.last_index
            self.labels.append(value)
            self.last_index += 1

            return self.encoding[value]

    def transform(self, values):
        return [self.encoding[x] for x in values]
    
    def load(self, filepath):
        with open(filepath, 'r') as _file:
            checkpoint = load(_file)
        
        self.encoding    = checkpoint['encoding']
        self.labels      = checkpoint['labels']
        self.last_index  = checkpoint['last_index']
    
    def save(self, name):
        checkpoint = {
            'encoding'      : self.encoding,
            'labels'        : self.labels,
            'last_index'    : self.last_index
        }

        with open(f'{name}.json', 'w+') as _file:
            dump(checkpoint, _file)
