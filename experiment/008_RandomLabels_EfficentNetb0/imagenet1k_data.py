import torchvision
import numpy as np

def IMAGENET1KRandomLabels(root, 
                           train: bool,
                           transform,
                           corrupt_prob):
    if train:
        split = 'train'
    else:
        split = 'val'
    
    dataset = torchvision.datasets.ImageNet(root, 
                                            split='val', 
                                            transform=transform)
    
    if corrupt_prob > 0:
      dataset.targets = _corrupt_labels(corrupt_prob, dataset=dataset)

    return dataset

def _corrupt_labels(corrupt_prob, dataset):
    num_labels = 1000
    labels = np.array(dataset.targets)

    ### Pluskid's Code
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(num_labels, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    return labels

    
