import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data import LakeDataset
from src.models import build_model
from src.metrics import scores

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def run_epoch(model,loader,device,optimizer=None):
    train=optimizer is not None; model.train(train)
    loss_sum=0.; n=0; stats=dict(tp=0,tn=0,fp=0,fn=0)
    with torch.set_grad_enabled(train):
        for batch in loader:
            x=batch['image'].to(device); y=batch['mask'].to(device)
            logits=model(x)
            bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
            prob=torch.sigmoid(logits)
            dice_loss=1-(2*(prob*y).sum(dim=(1,2,3))+1)/((prob+y).sum(dim=(1,2,3))+1)
            loss=bce+dice_loss.mean()
            if train:
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            n+=len(x);loss_sum+=loss.item()*len(x)
            s=scores(y.detach().cpu().numpy(),prob.detach().cpu().numpy())
            for key in stats: stats[key]+=s[key]
    tp,tn,fp,fn=(stats[k] for k in ('tp','tn','fp','fn'))
    return dict(loss=loss_sum/n,dice=2*tp/(2*tp+fp+fn+1e-8),iou=tp/(tp+fp+fn+1e-8),precision=tp/(tp+fp+1e-8),recall=tp/(tp+fn+1e-8))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True);ap.add_argument('--model',choices=['unet','deeplabv3plus','segformer_b0'],default='unet')
    ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--patience',type=int,default=7);ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--size',type=int,default=352);ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--out',default='results/unet_run')
    args=ap.parse_args();seed_all(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device!='cuda': raise RuntimeError('GPU not available; check allocation and CUDA')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tr=DataLoader(LakeDataset(args.manifest,'train',args.size,True),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    va=DataLoader(LakeDataset(args.manifest,'val',args.size),batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
    model=build_model(args.model).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    history=[];best=-1;bad=0
    for ep in range(1,args.epochs+1):
        train=run_epoch(model,tr,device,opt);val=run_epoch(model,va,device)
        row={'epoch':ep,**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()}}
        history.append(row)
        with open(out/'history.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerows(history)
        print(json.dumps(row),flush=True)
        if val['dice']>best+1e-5:
            best=val['dice'];bad=0
            torch.save({'state':model.state_dict(),'architecture':args.model,'size':args.size,'seed':args.seed,'best_epoch':ep,'best_val_dice':best,'manifest':args.manifest},out/'best.pt')
        else: bad+=1
        if bad>=args.patience: print('Early stopping',flush=True);break
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['state']);print('Best checkpoint restored',ck['best_epoch'],ck['best_val_dice'])
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for metric in ('loss','dice','iou'):
            plt.figure();plt.plot([r['epoch'] for r in history],[r['train_'+metric] for r in history],label='train');plt.plot([r['epoch'] for r in history],[r['val_'+metric] for r in history],label='val');plt.xlabel('Epoch');plt.ylabel(metric);plt.legend();plt.tight_layout();plt.savefig(out/f'{metric}_curve.png',dpi=250);plt.close()
    except ImportError: pass
if __name__=='__main__': main()
