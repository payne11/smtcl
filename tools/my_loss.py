import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        #首先对inputs每一个元素平方，在这里inputs为二维张量，所以dim=1就是在每一个向量里面累和。
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        #得到大矩阵[[|p0-p0|,...,|p0-p63|],...,[|p63-p0|,...,|p63-p63|]];
        # For each anchor, find the hardest positive and negative
        #每一行表示如果此样本的类别与其他样本相同则为1否则为0
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            #找出每一行同类样本的distance的最大值。
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            #找出每一行不同类样本distance的最小值。由[]->[1, ]
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        #返回填充了标量值 1 的张量，大小与输入相同。
        y = torch.ones_like(dist_an)
        # Compute ranking hinge loss
        #loss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)
        return self.ranking_loss(dist_an, dist_ap, y)

class CenterLoss(nn.Module):
    def __init__(self, num_classes=40, feat_dim=1024, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):

        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


class Triplet_Center_Loss(nn.Module):

    def __init__(self, num_classes=40, feat_dim=1024,margin=1, use_gpu=True):
        super(Triplet_Center_Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.margin=margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        if self.use_gpu:
            centers = torch.randn(self.num_classes, self.feat_dim).cuda()
        else:
            centers = torch.randn(self.num_classes, self.feat_dim)
        self.centers=nn.Parameter(centers)

    def forward(self, x, labels):

        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        #center[40,1024]
        #self.centers.data=F.normalize(self.centers,p=2,dim=1)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        distmat= distmat.clamp(min=1e-12).sqrt()
        # 得到大矩阵[[|p0-c0|,...,|p0-c40|],...,[|p63-c0|,...,|p63-c40|]];
        # 生成[0,1,...,39]
        classes = torch.arange(self.num_classes).long()#40
        if self.use_gpu: classes = classes.cuda()
        #由[64]->[64,1]相当于列向量->[64,40]
        class_1=classes.expand(batch_size, self.num_classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)#横向拉伸
        mask = labels.eq(classes.expand(batch_size, self.num_classes))#将[40]复制64份
        mask_not_eq=1-labels.eq(classes.expand(batch_size, self.num_classes))

        dist_ap=[]
        dist_an_nearest=[]
        for i in range(batch_size):
            dist_ap.append(distmat[i][mask[i]])#样本a与同类样本中心的距离
            #d0=distmat[i][mask[i]]#tensor([])
            #d1=distmat[i][mask_not_eq[i]].min().unsqueeze(0)#这里就变成tensor([])
            #d2=distmat[i][mask_not_eq[i]].min()#这是一个整数
            dist_an_nearest.append(distmat[i][mask_not_eq[i]].min().unsqueeze(0))#样本a与不同类样本距离最近中心的距离
        dist_ap = torch.cat(dist_ap)
        dist_an_nearest = torch.cat(dist_an_nearest)
        dis_ap_an=dist_ap-dist_an_nearest
        y = torch.ones_like(dist_an_nearest)
        # loss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)
        # loss(x1,x2)=sigmoid(5(dp-dn))*|dp-dn|
        # (1.0/(1+np.exp(-5(dp-dn))))*abs(dp-dn)
        loss=((1.0/(1+torch.exp(-1*(dis_ap_an))))*torch.abs(dis_ap_an)).sum()/len(dis_ap_an)
        return loss,dis_ap_an   #loss   self.ranking_loss(dist_an_nearest, dist_ap, y)


class soft_margin_triplet(nn.Module):
    def __init__(self,momentum=0.9,max_dist=None,nbins=64):
        super(soft_margin_triplet,self).__init__()
        if max_dist is None:
            max_dist = 2.0
        self._stats_initialized = False
        self._momentum=momentum
        self._max_val=max_dist
        self._min_val=-max_dist
        self.register_buffer("histogram",torch.ones(nbins))#self.register_buffer可以将tensor注册成buffer,buffer的更新在forward中，optim.step只能更新nn.Parameter类型的参数
    def forward(self, x, targets):

        n = x.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, x, x.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        zzz=targets.expand(n, n)
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        loss_sum=0
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        pos_dist = torch.cat(dist_ap)
        neg_dist= torch.cat(dist_an)

        #pos_dist,neg_dist=self.find_hard_negatives(dist,empirical_thresh=0.008)
        hist_val = pos_dist - neg_dist
        self._compute_stats(pos_dist,neg_dist)

        PDF=self.histogram/self.histogram.sum()
        CDF=PDF.cumsum(0)#进行累加

        # lookup weight from the CDF
        bin_idx = torch.floor((hist_val - self._min_val) / self.bin_width).long()
        weight = CDF[bin_idx].cuda()

        loss = -(neg_dist * weight).mean() + (pos_dist * weight).mean()
        return loss


    def _compute_stats(self, pos_dist, neg_dist):
        hist_val = pos_dist - neg_dist
        if self._stats_initialized:
            self._compute_histogram(hist_val, self._momentum)
        else:
            self._compute_histogram(hist_val, 1.0)
            self._stats_initialized = True
    def _compute_histogram(self, x, momentum):
        """
        update the histogram using the current batch
        """
        zzz=torch.max(x).item()
        max_val=torch.max(x).item()
        min_val=torch.min(x).item()
        if max_val>self._max_val:
            self._max_val=max_val
        if min_val<self._min_val:
            self._min_val=min_val

        num_bins = self.histogram.size(0)
        x_detached = x.detach()
        self.bin_width = (self._max_val - self._min_val) / (num_bins - 1)
        lo = torch.floor((x_detached - self._min_val) / self.bin_width).long()
        hi = (lo + 1).clamp(min=0, max=num_bins - 1)
        hist = x.new_zeros(num_bins)
        alpha = (1.0- (x_detached - self._min_val - lo.float() * self.bin_width)/self.bin_width)
        hist.index_add_(0, lo, alpha)
        hist.index_add_(0, hi, 1.0 - alpha)
        hist = (hist / (hist.sum() + 1e-6)).cpu()
        self.histogram = (1.0 - momentum) * self.histogram + momentum * hist


class improved_soft_Triplet_Center_Loss(nn.Module):

    def __init__(self, num_classes=40, feat_dim=1024, use_gpu=True):
        super(improved_soft_Triplet_Center_Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            centers = torch.randn(self.num_classes, self.feat_dim).cuda()
        else:
            centers = torch.randn(self.num_classes, self.feat_dim)
        self.centers=nn.Parameter(centers)
        self.ranking_loss = nn.MarginRankingLoss(margin=0)
    def forward(self, x, labels):

        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        #center[40,1024]
        self.centers.data=F.normalize(self.centers,p=2,dim=1)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        # 得到大矩阵[[|p0-c0|,...,|p0-c40|],...,[|p63-c0|,...,|p63-c40|]];
        # 生成[0,1,...,39]
        classes = torch.arange(self.num_classes).long()#40
        if self.use_gpu: classes = classes.cuda()
        #由[64]->[64,1]相当于列向量->[64,40]
        class_1=classes.expand(batch_size, self.num_classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)#横向拉伸
        mask = labels.eq(classes.expand(batch_size, self.num_classes))#将[40]复制64份
        mask_not_eq=1-labels.eq(classes.expand(batch_size, self.num_classes))

        dist_ap=[]
        dist_an_nearest=[]
        for i in range(batch_size):
            dist_ap.append(distmat[i][mask[i]])
            #d0=distmat[i][mask[i]]#tensor([])
            #d1=distmat[i][mask_not_eq[i]].min().unsqueeze(0)#这里就变成tensor([])
            #d2=distmat[i][mask_not_eq[i]].min()#这是一个整数
            dist_an_nearest.append(distmat[i][mask_not_eq[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an_nearest = torch.cat(dist_an_nearest)

        dp_n=dist_ap-dist_an_nearest
        weight=torch.exp(dp_n)
        weight=F.normalize(weight, p=2, dim=1)
        loss=self.ranking_loss(dist_an_nearest, dist_ap, weight)
        return loss

class soft_margin_triplet_centor_loss(nn.Module):
    def __init__(self,momentum=0.9,max_dist=None,nbins=64,use_gpu=True,num_classes=40,feat_dim=1024):
        super(soft_margin_triplet_centor_loss,self).__init__()
        if max_dist is None:
            max_dist = 2.0
        self._stats_initialized = False
        self._momentum=momentum
        self._max_val=max_dist
        self._min_val=-max_dist
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.register_buffer("histogram",torch.ones(nbins))#self.register_buffer可以将tensor注册成buffer,buffer的更新在forward中，optim.step只能更新nn.Parameter类型的参数
        if self.use_gpu:
            centers = torch.randn(self.num_classes, self.feat_dim).cuda()
        else:
            centers = torch.randn(self.num_classes, self.feat_dim)
        self.centers=nn.Parameter(centers)

    def forward(self, x, targets):
        batch_size = x.size(0)
        n = x.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        dist.addmm_(1, -2, x, self.centers.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative

        classes = torch.arange(self.num_classes).long()#40
        if self.use_gpu: classes = classes.cuda()


        labels = targets.unsqueeze(1).expand(batch_size, self.num_classes)#横向拉伸
        mask = labels.eq(classes.expand(batch_size, self.num_classes))#将[40]复制64份
        mask_not_eq=1-labels.eq(classes.expand(batch_size, self.num_classes))

        dist_ap=[]
        dist_an_nearest=[]
        for i in range(batch_size):
            dist_ap.append(dist[i][mask[i]])#样本a与同类样本中心的距离
            dist_an_nearest.append(dist[i][mask_not_eq[i]].min().unsqueeze(0))#样本a与不同类样本距离最近中心的距离
        pos_dist = torch.cat(dist_ap)
        neg_dist= torch.cat(dist_an_nearest)

        #pos_dist,neg_dist=self.find_hard_negatives(dist,empirical_thresh=0.008)
        hist_val = pos_dist - neg_dist
        self._compute_stats(pos_dist,neg_dist)

        PDF=self.histogram/self.histogram.sum()
        CDF=PDF.cumsum(0)#进行累加

        # lookup weight from the CDF
        bin_idx = torch.floor((hist_val - self._min_val) / self.bin_width).long()
        weight = CDF[bin_idx].cuda()

        loss = -(neg_dist * weight).mean() + (pos_dist * weight).mean()
        return loss


    def _compute_stats(self, pos_dist, neg_dist):
        hist_val = pos_dist - neg_dist
        if self._stats_initialized:
            self._compute_histogram(hist_val, self._momentum)
        else:
            self._compute_histogram(hist_val, 1.0)
            self._stats_initialized = True
    def _compute_histogram(self, x, momentum):
        """
        update the histogram using the current batch
        """
        zzz=torch.max(x).item()
        max_val=torch.max(x).item()
        min_val=torch.min(x).item()
        if max_val>self._max_val:
            self._max_val=max_val
        if min_val<self._min_val:
            self._min_val=min_val

        num_bins = self.histogram.size(0)
        x_detached = x.detach()
        self.bin_width = (self._max_val - self._min_val) / (num_bins - 1)
        lo = torch.floor((x_detached - self._min_val) / self.bin_width).long()
        hi = (lo + 1).clamp(min=0, max=num_bins - 1)
        hist = x.new_zeros(num_bins)
        alpha = (1.0- (x_detached - self._min_val - lo.float() * self.bin_width)/self.bin_width)
        hist.index_add_(0, lo, alpha)
        hist.index_add_(0, hi, 1.0 - alpha)
        hist = (hist / (hist.sum() + 1e-6)).cpu()
        self.histogram = (1.0 - momentum) * self.histogram + momentum * hist

