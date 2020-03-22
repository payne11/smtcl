""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

def map_and_auc_np(label, d):
    rs = convert_rank_gt(label, d)
    trec_precisions = []
    mrecs = []
    mpres = []
    aps = []
    for i, r in enumerate(rs):
        res = precision_and_recall(rs[i])
        trec_precisions.append(res[0])
        mrecs.append(res[1])
        mpres.append(res[2])
        aps.append(res[3])

    trec_precisions = np.stack(trec_precisions)
    mrecs = np.stack(mrecs)
    mpres = np.stack(mpres)
    aps = np.stack(aps)
    AUC = np.mean(aps)
    mAP = np.mean(trec_precisions)
    return AUC, mAP


def map_and_auc(label, d,top_k):
    idx = d.argsort(dim=1)
    gt = label.view(label.shape[0], 1) == label.view(1, label.shape[0])
    rs = [gt[i][idx[i]] for i in range(gt.shape[0])]

    trec_precisions = []
    aps = []
    for r in rs:
        r=r[:top_k]
        num_gt = torch.sum(r)
        trec_precision = torch.Tensor([torch.mean(r[:i+1].float()) for i in range(r.shape[0]) if r[i]])
        recall = [torch.sum(r[:i + 1].float()) / num_gt for i in range(r.shape[0])]
        precision = [torch.mean(r[:i + 1].float()) for i in range(r.shape[0])]

        # interpolate it
        mrec = torch.Tensor([0.] + recall + [1.])
        mpre = torch.Tensor([0.] + precision + [0.])

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = torch.max(mpre[i], mpre[i + 1])

        i = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = torch.sum((mrec[i] - mrec[i - 1]) * mpre[i])

        trec_precisions.append(trec_precision)
        aps.append(ap)

    trec_precisions = torch.cat(trec_precisions)
    aps = torch.stack(aps)
    AUC = torch.mean(aps)
    mAP = torch.mean(trec_precisions)
    return AUC.item(), mAP.item()


def convert_rank_gt(label, d):
    idx = d.argsort(axis=1)#返回每一行从小到大的index axis=1按行排序，axis=0按列进行排序
    gt = (label.reshape(label.size, 1) == label.reshape(1, label.size))#由n形成(n,n)矩阵数值非0即1，每行中为1的表示同类，0表示非同类
    rs = [gt[i][idx[i]] for i in range(gt.shape[0])] # rank ground truth gt.shape[0]表示每行的数值个数。从小到大排序，因为数值为距离，距离越小，分数越高。
    return rs


def precision_and_recall(r):
    num_gt = np.sum(r)
    trec_precision = np.array([np.mean(r[:i+1]) for i in range(r.size) if r[i]])
    recall = [np.sum(r[:i+1])/num_gt for i in range(r.size)]
    precision = [np.mean(r[:i+1]) for i in range(r.size)]

    # interpolate it
    mrec = np.array([0.] + recall + [1.])
    mpre = np.array([0.] + precision + [0.])

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i = np.where(mrec[1:] != mrec[:-1])[0]+1
    ap = np.sum((mrec[i]-mrec[i-1]) * mpre[i])
    return trec_precision, mrec, mpre, ap


def plot_pr_cure(mpres, mrecs):
    pr_curve = np.zeros(mpres.shape[0], 10)
    for r in range(mpres.shape[0]):
        this_mprec = mpres[r]
        for c in range(10):
            pr_curve[r, c] = np.max(this_mprec[mrecs[r]>(c-1)*0.1])
    return pr_curve
def dist_np(inputs):
    n = inputs.shape[0]
    # Compute pairwise distance, replace by the official when merged
    # 首先对inputs每一个元素平方，在这里inputs为二维张量，所以dim=1就是在每一个向量里面累和。
    dist = pow(inputs, 2).sum(axis=1, keepdims=True)

    dist = dist + dist.T
    inp=-2*np.dot(inputs,inputs.T)
    dist=dist+inp
    # dist.addmm_(1, -2, inputs, inputs.t())
    dist = np.sqrt(dist)  # for numerical stability
    dist=np.nan_to_num(dist)
    return dist
def dist(inputs):
    n = inputs.size(0)
    # Compute pairwise distance, replace by the official when merged
    # 首先对inputs每一个元素平方，在这里inputs为二维张量，所以dim=1就是在每一个向量里面累和。
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist =torch.sqrt(dist)  # for numerical stability
    return dist