import numpy as np

def scores(y, p, threshold=.5):
    y=np.asarray(y)>0.5; pred=np.asarray(p)>=threshold
    tp=int(np.logical_and(y,pred).sum()); tn=int(np.logical_and(~y,~pred).sum())
    fp=int(np.logical_and(~y,pred).sum()); fn=int(np.logical_and(y,~pred).sum())
    return dict(dice=(2*tp)/(2*tp+fp+fn+1e-8),iou=tp/(tp+fp+fn+1e-8),precision=tp/(tp+fp+1e-8),recall=tp/(tp+fn+1e-8),specificity=tn/(tn+fp+1e-8),accuracy=(tp+tn)/(tp+tn+fp+fn+1e-8),tp=tp,tn=tn,fp=fp,fn=fn)
