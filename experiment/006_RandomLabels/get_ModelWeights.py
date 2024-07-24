import sys
sys.path.append('/scratch/cm6627/diffeo_cnn/experiment/006_RandomLabels/fitting-random-labels')

import torch
import model_wideresnet 

class ModelWeights:
    
    path = '/scratch/cm6627/diffeo_cnn/experiment/006_RandomLabels/ModelWeights/'
    EpochsAmount = [0, 60, 120, 180, 240]
    CorruptAmount = [0.0, 0.5, 1.0]

    @staticmethod
    def load_Model(corrupt: float, epochs: int) -> 'torch.model':
        """
        Loads WideResNet w/ weights trained w/ various amounts 
        of corruption and at different epochs.

        Args:
        - corrupt (float): Percentage (in decimal form) of labels randomized
        - epochs (int): Number of epochs during training
        
        Example usage:
        ```
        model = ModelWeights.load_Model(corrupt=0.5, epochs=180)
        ```
        """
        ### Checks
        if corrupt not in ModelWeights.CorruptAmount:
            raise ValueError(f'`corrupt` must be: {ModelWeights.CorruptAmount}')
        if epochs not in ModelWeights.EpochsAmount:
            raise ValueError(f'`epochs` must be {ModelWeights.EpochsAmount}')

        ### Code
        epochs = str(int(epochs))
        if corrupt == 0.0:
            corrupt = '0p0'
        elif corrupt == 0.5:
            corrupt = '0p5'
        elif corrupt == 1.0:
            corrupt = '1p0'

        if epochs == 0:  # This is just a randomly initalized model 
            corrupt = '0p0'

        file_name = f'/Corrupt-{corrupt}/ModelWeights_{epochs}Epochs.pth'

        # I trained on these parameters, which are default to the paper's code
        depth = 28
        classes = 10
        widen_factor = 1
        drop_rate = 0
        model = model_wideresnet.WideResNet(depth, classes,
                                            widen_factor,
                                            drop_rate=drop_rate)

        model_weights_path = ModelWeights.path + file_name
        model.load_state_dict(torch.load(model_weights_path))
        
        return model
