import argparse,csv,random
from pathlib import Path

EXT={'.png','.jpg','.jpeg','.tif','.tiff'}
def main():
    p=argparse.ArgumentParser(description='Build CSV from matched image/mask stems; use curated region-isolated folders for test split.')
    p.add_argument('--root',required=True,help='Directory with train/images, train/masks, val/images, val/masks, optionally test/...')
    p.add_argument('--out',default='configs/manifest.csv');p.add_argument('--require-test',action='store_true');a=p.parse_args()
    root=Path(a.root); rows=[]
    splits=('train','val','test') if a.require_test else ('train','val')
    for split in splits:
        image_dir=root/split/'images';mask_dir=root/split/'masks'
        if not image_dir.is_dir() or not mask_dir.is_dir():raise SystemExit(f'Missing folders for {split}: {image_dir} / {mask_dir}')
        images={p.stem:p for p in image_dir.rglob('*') if p.suffix.lower() in EXT}
        masks={p.stem:p for p in mask_dir.rglob('*') if p.suffix.lower() in EXT}
        missing=images.keys()-masks.keys()
        if missing: raise SystemExit(f'{split}: {len(missing)} images have no same-stem mask; examples {list(sorted(missing))[:8]}')
        if len(images)!=len(masks):print(f'Warning: {split} has {len(masks)-len(images)} unmatched masks')
        for stem,im in sorted(images.items()):rows.append(dict(image_path=str(im.resolve()),mask_path=str(masks[stem].resolve()),split=split,region='unknown'))
        print(split,len(images),'matched pairs')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['image_path','mask_path','split','region']);w.writeheader();w.writerows(rows)
    print('Saved',out)
if __name__=='__main__':main()
