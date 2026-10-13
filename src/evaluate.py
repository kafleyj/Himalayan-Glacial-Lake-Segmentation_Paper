import argparse,json,csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from src.models import build_model
from src.data import LakeDataset
from src.metrics import scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--split',default='test');ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--out',default='results/test');ap.add_argument('--save-samples',type=int,default=40);a=ap.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu';ck=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model=build_model(ck['architecture']).to(device);model.load_state_dict(ck['state'],strict=True);model.eval()
    loader=DataLoader(LakeDataset(a.manifest,a.split,ck['size']),batch_size=a.batch_size,num_workers=1)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'predictions').mkdir(exist_ok=True)
    totals=dict(tp=0,tn=0,fp=0,fn=0);rows=[];saved=0
    with torch.inference_mode():
        for batch in loader:
            prob=torch.sigmoid(model(batch['image'].to(device))).cpu().numpy()
            gt=batch['mask'].numpy()
            for i,path in enumerate(batch['path']):
                s=scores(gt[i],prob[i]);rows.append(dict(path=path,region=batch['region'][i],**s))
                for k in totals: totals[k]+=s[k]
                if saved<a.save_samples:
                    pred=(prob[i,0]>=.5).astype(np.uint8)*255
                    Image.fromarray(pred).save(out/'predictions'/f'{saved:04d}_mask.png');saved+=1
    tp,tn,fp,fn=(totals[k] for k in ('tp','tn','fp','fn'))
    met=dict(n=len(rows),**totals,dice=2*tp/(2*tp+fp+fn+1e-8),iou=tp/(tp+fp+fn+1e-8),precision=tp/(tp+fp+1e-8),recall=tp/(tp+fn+1e-8),specificity=tn/(tn+fp+1e-8),accuracy=(tp+tn)/(tp+tn+fp+fn+1e-8))
    (out/'metrics.json').write_text(json.dumps(met,indent=2))
    with open(out/'image_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps(met,indent=2))
if __name__=='__main__':main()
