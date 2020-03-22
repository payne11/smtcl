import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import time
import pandas as pd
import utils
class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, optimizer_centor,loss_fn_softmax,loss_fn_center, \
                 model_name, log_dir, num_views=12):

        self.optimizer = optimizer
        self.optimizer_centor = optimizer_centor
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn_softmax = loss_fn_softmax
        self.loss_fn_center = loss_fn_center
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)


    def train(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()
        weight_cent=1;
        xent_losses = AverageMeter()
        cent_losses = AverageMeter()
        losses = AverageMeter()
        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)
            dis_an_ap=[]
            # train one epoch
            for i, data in enumerate(self.train_loader):

                if self.model_name == 'mvcnn':
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1,C,H,W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()


                out_feature,out_data = self.model(in_data)
                confid_score=nn.Softmax(out_data)
                loss_cent,dis_np=self.loss_fn_center(out_feature, target)
                dis_np=dis_np.detach().cpu().numpy()
                dis_an_ap.append(dis_np)
                loss_xcent=self.loss_fn_softmax(out_data,target)
                loss_cent*=weight_cent
                loss=loss_xcent+loss_cent

                #log_str_dist_an_ap = 'epoch %d,step %d: dist_an_ap %s;' % (epoch + 1 , i+1, dis_np)
                #data = open("dist_an_ap.txt", 'a')
                #print(log_str_dist_an_ap, file=data)
                #data.close()

                self.optimizer.zero_grad()
                self.optimizer_centor.zero_grad()
                self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

                #pred = torch.max(out_data, 1)[1]
                pred_scores, pred = torch.max(out_data, 1)
                results = pred == target
                correct_points = torch.sum(results.long())
                acc = correct_points.float()/results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc+i+1)

                loss.backward()
                self.optimizer.step()

                for param in self.loss_fn_center.parameters():
                    param.grad.data *= (1. / weight_cent)
                self.optimizer_centor.step()

                losses.update(loss.item(),target.size(0));
                xent_losses.update(loss_xcent.item(),target.size(0))
                cent_losses.update(loss_cent.item(),target.size(0))
                
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch+1, i+1, loss, acc)
                if (i+1)%1==0:
                    print(log_str)





            i_acc += i
            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc,mAP,AUC= self.update_validation_accuracy(epoch)

                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
                self.writer.add_scalar('val/val_loss', loss, epoch+1)
                self.writer.add_scalar('val/mAP', mAP, epoch + 1)
                self.writer.add_scalar('val/AUC', AUC, epoch + 1)
            #log_str_mAP = 'epoch %d, step %d: val_loss %.3f; val_acc %.3f;val_mAP  %.3f;val_auc_micro  %.3f;val_auc_macro  %.3f;' % (epoch + 1, i + 1, loss, acc,mAP,auc_micro,auc_macro)

            # data = open( "mAP.txt", 'a')
            # print(log_str_mAP, file=data)
            # data.close()
            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)
 
            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir+"/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0
        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred_scores = []
        out_f=[]
        with torch.no_grad():
            for _, data in enumerate(self.val_loader, 0):

                if self.model_name == 'mvcnn':
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:  # 'svcnn'
                    in_data = Variable(data[1]).cuda()
                target = Variable(data[0]).cuda()

                out_feature, out_data = self.model(in_data)
                # out_feature.cpu().detach().numpy()
                # out_f.append(out_feature.detach())
                # out_f.extend(out_feature.cpu().detach().numpy())
                pred_scores, pred = torch.max(out_data, 1)
                # all_loss += 0.01*self.loss_fn_1(out_feature, target).cpu().data.numpy()+self.loss_fn_2(out_data, target).cpu().data.numpy()#
                loss_cent, dis_an_ap = self.loss_fn_center(out_feature, target)
                loss_xcent = self.loss_fn_softmax(out_data, target)
                loss_cent *= 1
                loss = loss_xcent + loss_cent  # +
                all_loss += loss
                results = pred == target
                arr_pred_scores = out_data.cpu().detach().numpy()
                all_pred_scores.extend(arr_pred_scores)
                arr_target = target.cpu().detach().numpy()
                all_target.extend(arr_target)
                # all_target.extend(target.detach())
                for i in range(results.size()[0]):
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                    samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]
        # #calc mAP and AUC
        # all_target=torch.tensor(all_target)
        # out_f=np.array(out_f)
        # # out_f=torch.cat(out_f)
        # distance=utils.dist_np(out_f)
        # res=[]
        # for i in range(distance.shape[0]):
        #     res.append(torch.tensor(distance[i]).view(-1,distance[i].shape[0]))
        # distance=torch.cat(res)
        # # AUC,mAP=utils.map_and_auc_np(all_target,distance)
        # AUC,mAP=utils.map_and_auc(all_target,distance,top_k=top)

        ap = 0
        auc= 0
        # y_label=pd.DataFrame(all_target)
        # y_one_hot = label_binarize(y_label, np.arange(40))
        # auc_micro=metrics.roc_auc_score(y_one_hot, all_pred_scores, average='micro')
        # auc_macro = metrics.roc_auc_score(y_one_hot, all_pred_scores, average='macro')
        for label in range(40):
            label_target = [sample_labels == label for sample_labels in all_target]
            label_target = np.array(label_target)
            label_target = label_target.astype(int)
            table = []
            label_socre = [x[label] for x in all_pred_scores]
            # arr_target=target.cpu().detach().numpy()
            for target_size in range(len(all_target)):
                table.append([label_socre[target_size], label_target[target_size]])
            table = np.array(table)
            each_ap,each_auc = self.calcu_map(table)
            ap=each_ap+ap
            auc=each_auc+auc
        mAP = ap / 40
        AUC = auc/ 40
        print ('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)
        print ('val mAP : ', mAP)
        print ('val AUC : ', AUC)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc,mAP,AUC

    def calcu_map(self,table):
        #按照第0列，argsort()函数是将x中的元素从小到大排列，[::-1]取从后向前的元素
        index=np.argsort(table[:,0])[::-1]
        T=table[index]#[置信度得分，类别]
        length=T.shape[0]
        TP=len(T[T[:,-1]>0])

        avg_precision = []
        avg_precision.extend(np.mean(T[:i + 1, 1]) for i in range(T.shape[0]) if T[i, 1])
        avg_precision = np.array(avg_precision)
        ap = np.mean(avg_precision)

        if TP==0:
            return 0
        result=[]
        # 计算top-N的 召回率 与 准确率
        for i in range(length):
            temp=np.sum(T[:i+1,-1]>0)
            result.append([temp/TP,temp/(i+1)])#0为recall,1为precision
        result1=np.array(result)
        result2=[]

        # 计算每一个召回率对应的最大的准确率
        for i in range(TP):
            result2.append([(i + 1) / TP, np.max(result1[result1[:, 0] == (i + 1) / TP][:, 1])])
        result2=np.array(result2)

        result3=[]

        for i in range(result2.shape[0]):
            result3.append([result2[i, 0], np.max(result2[i:result2.shape[0], 1])])

        result3=np.array(result3)
        auc=np.mean(result3[:,1])
        return ap,auc


class ModelNetTrainer_oneloss(object):

    def __init__(self, model, train_loader, val_loader, optimizer,  loss_fn,  \
                 model_name, log_dir, num_views=12):

        self.optimizer = optimizer

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)
    def train(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()

        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths) / self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[
                                     rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            for i, data in enumerate(self.train_loader):

                if self.model_name == 'mvcnn':
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                out_feature, out_data = self.model(in_data)
                loss = self.loss_fn(out_feature, target)

                self.optimizer.zero_grad()

                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)

                # pred = torch.max(out_data, 1)[1]
                pred_scores, pred = torch.max(out_data, 1)
                results = pred == target
                correct_points = torch.sum(results.long())
                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)

                loss.backward()
                self.optimizer.step()

                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 1 == 0:
                    print(log_str)

            i_acc += i
            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc, mAP, auc_micro, auc_macro = self.update_validation_accuracy(
                        epoch)

                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)
                self.writer.add_scalar('val/mAP', mAP, epoch + 1)
                self.writer.add_scalar('val/auc_micro', auc_micro, epoch + 1)
                self.writer.add_scalar('val/auc_macro', auc_macro, epoch + 1)
            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)
            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0
        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0
        self.model.eval()

        all_target = []
        all_pred_scores = []

        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'mvcnn':
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            else:  # 'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_feature, out_data = self.model(in_data)

            pred_scores, pred = torch.max(out_data, 1)
            all_loss += self.loss_fn(out_feature, target).cpu().data.numpy()
            results = pred == target

            arr_pred_scores = out_data.cpu().detach().numpy()
            all_pred_scores.extend(arr_pred_scores)
            arr_target = target.cpu().detach().numpy()
            all_target.extend(arr_target)
            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]
        ap = 0
        y_label = pd.DataFrame(all_target)
        y_one_hot = label_binarize(y_label, np.arange(40))
        auc_micro = metrics.roc_auc_score(y_one_hot, all_pred_scores, average='micro')
        auc_macro = metrics.roc_auc_score(y_one_hot, all_pred_scores, average='macro')
        for label in range(40):
            label_target = [sample_labels == label for sample_labels in all_target]
            label_target = np.array(label_target)
            label_target = label_target.astype(int)
            table = []
            label_socre = [x[label] for x in all_pred_scores]
            # arr_target=target.cpu().detach().numpy()
            for target_size in range(len(all_target)):
                table.append([label_socre[target_size], label_target[target_size]])
            table = np.array(table)
            ap += self.calcu_map(table)
        mAP = ap / 40
        print('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)
        print('val mAP : ', mAP)
        print('val auc_micro : ', auc_micro)
        print('val auc_macro : ', auc_macro)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc, mAP, auc_micro, auc_macro
    def calcu_map(self, table):
        # 按照第0列，argsort()函数是将x中的元素从小到大排列，[::-1]取从后向前的元素
        index = np.argsort(table[:, 0])[::-1]
        T = table[index]
        length = T.shape[0]
        TP = len(T[T[:, -1] > 0])

        if TP == 0:
            return 0
        result = []

        for i in range(length):
            temp = np.sum(T[:i + 1, -1] > 0)
            result.append([temp / TP, temp / (i + 1)])  # 0为recall,1为precision
        result1 = np.array(result)
        result2 = []

        for i in range(TP):
            result2.append([(i + 1) / TP, np.max(result1[result1[:, 0] == (i + 1) / TP][:, 1])])
        result2 = np.array(result2)

        result3 = []

        for i in range(result2.shape[0]):
            result3.append([result2[i, 0], np.max(result2[i:result2.shape[0], 1])])

        result3 = np.array(result3)
        ap = np.mean(result3[:, 1])
        return ap


def AUC(label, pre):


    # 计算正样本和负样本的索引，以便索引出之后的概率值
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]

    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5

    return auc / (len(pos) * len(neg))

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum +=val * n
        self.count += n
        self.avg = self.sum / self.count
