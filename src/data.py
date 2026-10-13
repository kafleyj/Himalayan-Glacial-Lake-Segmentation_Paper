import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class LakeDataset(Dataset):
    def __init__(self, manifest, split, size=352, augment=False):
        with open(manifest, newline='') as f:
            self.rows=[r for r in csv.DictReader(f) if r['split']==split]
        if not self.rows: raise ValueError(f'No rows for split {split} in {manifest}')
        self.size=size; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r=self.rows[idx]
        with Image.open(r['image_path']) as im: im=im.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR)
        with Image.open(r['mask_path']) as mk: mk=mk.convert('L').resize((self.size,self.size),Image.Resampling.NEAREST)
        x=np.array(im, dtype=np.float32)/255.
        y=(np.array(mk,dtype=np.uint8)>0).astype(np.float32)
        if self.augment:
            if torch.rand(())<.5: x=x[:,::-1].copy();y=y[:,::-1].copy()
            if torch.rand(())<.5: x=x[::-1].copy();y=y[::-1].copy()
            k=int(torch.randint(0,4,()).item()); x=np.rot90(x,k).copy();y=np.rot90(y,k).copy()
        x=(x-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        return {'image':torch.from_numpy(x.transpose(2,0,1).copy()),'mask':torch.from_numpy(y[None].copy()),'path':r['image_path'],'region':r.get('region','unknown')}
