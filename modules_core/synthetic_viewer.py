import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
sys.path.append('../')
from modules_core import dummy_loader

synthetic_dataset = dummy_loader.SyntheticNoiseDataset()
dataloader = DataLoader(synthetic_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['greyscale_image'].size(),
          sample_batched['augmented_image'].size())

    synthetic_dataset.show_images(**sample_batched)
    if i_batch == 3:
        break