import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.metrics import bbox_iou
import torchvision
from copy import deepcopy

"""YOLOv8 loss + L2 for LwF"""
class LwFLoss(v8DetectionLoss):
    def __init__(self, h, m, device, lwf=3.0, new_classes=[]):  # model must be de-paralleled
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.lwf = lwf
        self.new_classes = new_classes

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # LwF on the output
        self.lwf_loss = torch.nn.MSELoss()
        self.last_yolo_loss = 0
        self.last_lwf_loss = 0

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocesses the target counts and matches with the input batch size
        to output a tensor.
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """
        Decode predicted object bounding box coordinates from anchor points and
        distribution.
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, teacher_output):
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        teacher_output = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.clone().detach().sigmoid(),
            (pred_bboxes.clone().detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = (
               self.bce(pred_scores[:,:,self.new_classes], target_scores[:,:,self.new_classes].to(dtype)).sum() / target_scores_sum 
            )


        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor


            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        lwf_loss = 0
        filter_idx = self.reg_max * 4 + self.new_classes[0]

        for i in range(3):
            lwf_loss += self.lwf_loss(feats[i][:, : filter_idx,:,:], teacher_output[i][:, :filter_idx,:,:].detach())
        lwf_loss /= 3

        #print(type(lwf_loss))
        total_loss = loss.sum() * batch_size + self.lwf * lwf_loss * batch_size
        self.last_yolo_loss = loss.sum().item()
        self.last_lwf_loss = lwf_loss.item()

        #return torch.tensor(0.0, requires_grad=True), loss.detach()  # loss(box, cls, dfl)
        return total_loss, loss.detach()
    


"""YOLOv8 loss for YOLO LwF: YOLO loss + YOLO LwF loss"""
class LwFLossV2(v8DetectionLoss):
    def __init__(self, h, m, device, c1=1.0, c2=1.0, c3=None, old_classes=[], classes=None):  # model must be de-paralleled

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.classes = classes

        # custom LwF on the output
        self.lwf_loss = YOLOv8LwFLossNew(c1, c2, old_classes, m.reg_max, device, c3)
        #self.lwf_loss = ERS(c1, c2, old_classes, m.reg_max, device, c3)
        self.last_yolo_loss = 0
        self.last_lwf_loss = 0

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocesses the target counts and matches with the input batch size
        to output a tensor.
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """
        Decode predicted object bounding box coordinates from anchor points and
        distribution.
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, teacher_output):
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        teacher_output = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        lwf_loss = self.lwf_loss(pred_scores, pred_distri, teacher_output, anchor_points)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)



        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.clone().detach().sigmoid(),
            (pred_bboxes.clone().detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        
        # Cls loss
        if self.classes:
            loss[1] = (
                self.bce(pred_scores[:,:,self.classes], target_scores[:,:,self.classes].to(dtype)).sum() / target_scores_sum
            )  # BCE
        else:
            loss[1] = (
               self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum 
            )
        """
        loss[1] = (
               self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum 
            )
        """

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor


            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain


        self.last_yolo_loss = loss.sum().item()
        self.last_lwf_loss = lwf_loss.item()

        total_loss = loss.sum() * batch_size + lwf_loss * batch_size

        return total_loss, loss.detach()


"""YOLO LwF loss"""
class YOLOv8LwFLossNew(nn.Module):

    def __init__(self, c1, c2, classes, reg_max, device, c3=None):
        """
        :param c1: constant for classification LwF loss
        :param c2: constant for regression LwF loss
        :param classes: list of classes involved (classes for old tasks)
        :param c3: optional, constant for DFL
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.classes = classes
        self.c3 = c3

        self.reg_max = reg_max

        """
        Since each YOLOv8 output has shape [N, 4 * reg_max + nc, H, H] (N=batch size, H depends on the head), we consider the following reshapes:
        (following ultralytics code)
            1) [N, 4 * reg_max + nc, D], D=8400     (D = H_1 x H_1 + H_2 x H_2 + H_3 x H_3, H_i head dependent )
            2) split and permute -> [N, D, nc] and [N, D, 4 * reg_max] (for student, this is already done by ultralytics code)
            3) by following bbox_decode, use view to get [N, D, 4, reg_max]
        """

        """ Regression """
        self.log_softmax = nn.LogSoftmax(dim=3)  # for student
        self.softmax = nn.Softmax(dim=3)  # for teacher

        # -> cross entropy (sum along last dim)

        """Classification"""
        self.sigmoid = nn.Sigmoid()  # for teacher -> compute both target and weights (for regression)
        self.bce = nn.BCEWithLogitsLoss(reduce="none")

        """DFL utils"""
        self.proj = torch.arange(reg_max, dtype=torch.float, device=device)

    def __call__(self, student_cl_output, student_reg_output, teacher_output, anchors):
        """
        :param student_cl_output: [N, D, nc] tensor with cls outputs (logits) of student
        :param student_reg_output: [N, D, 4*reg_max] tensor with regression output of student
        :param teacher: list with outputs, one per head: [N, reg_max * 4 + nc, ...] or tensor [N, reg_max * 4 + nc, D]
        """

        reg_T = 2
        weights_T = 1

        # get number of classes and number of total outputs
        batch_size = student_cl_output.shape[0]  # N
        nc = student_cl_output.shape[-1]
        reg_max = student_reg_output.shape[-1] // 4
        no = reg_max * 4 + nc
        num_preds = student_cl_output.shape[1]  # D

        # reshape teacher output
        if isinstance(teacher_output, list):
            target_distri, target_logit_scores = torch.cat(
                [xi.view(batch_size, no, -1) for xi in teacher_output], 2
            ).split((reg_max * 4, nc), 1)
        else:
            target_distri, target_logit_scores = teacher_output.split((reg_max * 4, nc), 1)
        

        target_logit_scores = target_logit_scores.permute(0, 2, 1).contiguous()  # [N, D, 4*reg_max]
        target_distri = target_distri.permute(0, 2, 1).contiguous()  # [N, D, nc]

        IoU_scores = torch.pow(self.score_IoU(student_reg_output, target_distri, anchors).repeat(1,1,len(self.classes)), 1)
        #IoU_scores = torch.clip(torch.log(self.score_IoU(student_reg_output, target_distri, anchors).repeat(1,1,len(self.classes))) + 1, min=0.0)

        target_distri = target_distri.view(batch_size, num_preds, 4, reg_max)  # [N, D, 4, reg_max]
        target_scores = self.sigmoid(target_logit_scores)  # [N, D, nc]


        # reshape regression output for student
        pred_distri = student_reg_output.view(batch_size, num_preds, 4, reg_max)  # [N, D, 4, reg_max]

        """classification"""
        lwf_cls_loss = IoU_scores * self.bce(student_cl_output[:,:, self.classes], target_scores[:,:, self.classes].detach())

        lwf_cls_loss = torch.mean(lwf_cls_loss)

        """regression"""
        # compute weights
        #weights, _ = torch.max(target_scores[:, :, self.classes], dim=2)  # [N, D]
        weights, _ = torch.max((self.sigmoid(target_logit_scores / weights_T))[:, :, self.classes], dim=2 )

        # repeat
        weights = weights.unsqueeze(2).repeat(1,1,4)  # [N, D, 4]

        #entropies = self.entropy_score(target_distri)
        #entropies_scores = torch.pow(1 + entropies/math.log2(1/16), 1)

        # compute CEs
        target_distri = self.softmax(target_distri.detach() / reg_T)  # p
        log_pred_distri = self.log_softmax(pred_distri / reg_T)  # log q
        CEs = torch.sum( - target_distri * log_pred_distri, dim=3)  # [N, D, 4]

        # weight CEs
        weighted_CEs = torch.pow(weights, 1) * CEs

        lwf_regression_loss = torch.mean(weighted_CEs)

        lwf_loss = self.c1 * lwf_cls_loss + self.c2 * lwf_regression_loss
        #print(f"{ self.c1 * lwf_cls_loss}, {self.c2 * lwf_regression_loss}")

        """DFL"""
        if self.c3 is not None:
            target = target_distri.matmul(self.proj.type(target_distri.dtype))  # compute expected value for teacher [N, D, 4]
            df_loss = torch.mean(self._df_loss(log_pred_distri, target) * weights)
            lwf_loss += self.c3 * df_loss
            
        return lwf_loss


    def _df_loss(self, log_pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right

        tl_mask = F.one_hot(tl, self.reg_max).float()
        tr_mask = F.one_hot(tr, self.reg_max).float()

        left_term = - torch.sum(log_pred_dist * tl_mask, dim=-1) * wl
        right_term = - torch.sum(log_pred_dist * tr_mask, dim=-1) * wr

        return left_term + right_term
    
    def entropy_score(self, target_distri):

        log_probs = self.log_softmax(target_distri.detach())
        entropy = torch.sum(- self.softmax(target_distri.detach()) * log_probs, dim=-1)

        return entropy

    
    def score_IoU(self, pred_distri, target_distri, anchors):

        target = torch.unsqueeze(self.bbox_decode(anchors, target_distri), dim=-2)
        pred = torch.unsqueeze(self.bbox_decode(anchors, pred_distri.detach()), dim=-2)
        
        scores = (bbox_iou(pred, target, xywh=False, DIoU=True)+1)/2
        return torch.squeeze(scores, dim=-1)  # [N, D, 1]
    
    def bbox_decode(self, anchor_points, pred_dist):
        """
        Decode predicted object bounding box coordinates from anchor points and
        distribution.
        """
        
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    



class ERS(nn.Module):

    def __init__(self, c1, c2, classes, reg_max, device, c3=None):
        """
        :param c1: constant for classification LwF loss
        :param c2: constant for regression LwF loss
        :param classes: list of classes involved (classes for old tasks)
        :param c3: optional, constant for DFL
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.classes = classes
        self.c3 = c3

        self.reg_max = reg_max

        """
        Since each YOLOv8 output has shape [N, 4 * reg_max + nc, H, H] (N=batch size, H depends on the head), we consider the following reshapes:
        (following ultralytics code)
            1) [N, 4 * reg_max + nc, D], D=8400     (D = H_1 x H_1 + H_2 x H_2 + H_3 x H_3, H_i head dependent )
            2) split and permute -> [N, D, nc] and [N, D, 4 * reg_max] (for student, this is already done by ultralytics code)
            3) by following bbox_decode, use view to get [N, D, 4, reg_max]
        """

        """ Regression """
        self.log_softmax = nn.LogSoftmax(dim=3)  # for student
        self.softmax = nn.Softmax(dim=3)  # for teacher

        # -> cross entropy (sum along last dim)

        """Classification"""
        self.sigmoid = nn.Sigmoid()  # for teacher -> compute both target and weights (for regression)
        self.bce = nn.BCEWithLogitsLoss(reduce="none")

        """DFL utils"""
        self.proj = torch.arange(reg_max, dtype=torch.float, device=device)

    def __call__(self, student_cl_output, student_reg_output, teacher_output, anchors):
        """
        :param student_cl_output: [N, D, nc] tensor with cls outputs (logits) of student
        :param student_reg_output: [N, D, 4*reg_max] tensor with regression output of student
        :param teacher: list with outputs, one per head: [N, reg_max * 4 + nc, ...] or tensor [N, reg_max * 4 + nc, D]
        """
        cls_T = 1
        reg_T = 10

        # get number of classes and number of total outputs
        batch_size = student_cl_output.shape[0]  # N
        nc = student_cl_output.shape[-1]
        reg_max = student_reg_output.shape[-1] // 4
        no = reg_max * 4 + nc
        num_preds = student_cl_output.shape[1]  # D

        # reshape teacher output
        if isinstance(teacher_output, list):
            target_distri, target_logit_scores = torch.cat(
                [xi.view(batch_size, no, -1) for xi in teacher_output], 2
            ).split((reg_max * 4, nc), 1)
        else:
            target_distri, target_logit_scores = teacher_output.split((reg_max * 4, nc), 1)
        

        target_logit_scores = target_logit_scores.permute(0, 2, 1).contiguous()  # [N, D, nc]
        target_distri = target_distri.permute(0, 2, 1).contiguous()  # [N, D, 4*reg_max]

        target_distri = target_distri.view(batch_size, num_preds, 4, reg_max)  # [N, D, 4, reg_max], still logits
        target_scores = self.sigmoid(target_logit_scores / cls_T)  # [N, D, nc]

        # reshape regression output for student
        pred_distri = student_reg_output.view(batch_size, num_preds, 4, reg_max) # [N, D, 4, reg_max], logits

        """classification"""
        lwf_cls_loss = F.mse_loss(student_cl_output[:,:, self.classes], target_logit_scores[:,:, self.classes].detach(), reduce=None)

        use_cls = True # use ERS for classification
        use_cls_2 = False # second version

        if use_cls:
            
            if not use_cls_2:
                weights_cls = self.ers_cls_3(target_scores[:,:, self.classes].detach())
                lwf_cls_loss = torch.sum(torch.mean(weights_cls * lwf_cls_loss, dim=-1), dim=-1)
            else:
                weights_cls = self.ers_cls_2(target_scores[:,:, self.classes].detach())
                lwf_cls_loss = torch.sum(weights_cls * lwf_cls_loss, dim=(-2,-1))

        #lwf_cls_loss = torch.mean(lwf_cls_loss * weights_cls)
        lwf_cls_loss = torch.mean(lwf_cls_loss)

        """regression"""
        # compute weights
        target_distri_ = self.softmax(target_distri.detach())  # [N, D, 4, reg_max], P
        weights = self.ers_regression_2(target_distri_, target_scores, anchors)

        #weights = self.ers_regression(target_distri, target_scores, anchors)
        target_distri = self.softmax(target_distri.detach() / reg_T)
        log_pred_distri = self.log_softmax(pred_distri / reg_T)  # log q
        CEs = torch.sum( - target_distri * log_pred_distri, dim=3)  # [N, D, 4]

        # weight CEs (mask bad predictions based on ERS)
        weighted_CEs = weights * CEs

        # mean w.r.t. 4 offsets, sum w.r.t. all predictions and mean w.r.t. batch size
        lwf_regression_loss = torch.mean( torch.sum(torch.mean(weighted_CEs, dim=-1), dim=-1) )

        lwf_loss = self.c1 * lwf_cls_loss + self.c2 * lwf_regression_loss
        #print(f"{ self.c1 * lwf_cls_loss}, {self.c2 * lwf_regression_loss}")

        """DFL"""
        if self.c3 is not None:
            target = target_distri.matmul(self.proj.type(target_distri.dtype))  # compute expected value for teacher [N, D, 4]
            df_loss = torch.mean(self._df_loss(log_pred_distri, target) * weights)
            lwf_loss += self.c3 * df_loss
            
        return lwf_loss


    def _df_loss(self, log_pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right

        tl_mask = F.one_hot(tl, self.reg_max).float()
        tr_mask = F.one_hot(tr, self.reg_max).float()

        left_term = - torch.sum(log_pred_dist * tl_mask, dim=-1) * wl
        right_term = - torch.sum(log_pred_dist * tr_mask, dim=-1) * wr

        return left_term + right_term
    

    def ers_regression(self, teacher_output, teacher_scores, anchors):
        """
        :param: [N, D, 4, reg_max] tensor with reg outputs (probs) of teacher
        :param teacher_scores: [N, D, nc] tensor with cls output of teacher (after sigmoid)
        :param anchors: anchor points
        """

        # target_distri = target_distri.view(batch_size, num_preds, 4, reg_max)  # [N, D, 4, reg_max]
        #print(teacher_output.shape)
        alpha = 2
        D = teacher_output.shape[1]

        # top1 for each bbox
        top1 = torch.amax(teacher_output, dim=(-1,-2))  # [N, D]

        # get stats
        mean = torch.mean(top1, dim=-1)  # [N]
        std = torch.std(top1, dim=-1)

        # derive threshold
        thresholds = mean + alpha * std
        thresholds = thresholds.unsqueeze(-1).repeat(1,D)  # [N,D]

        # compute mask
        weights = (top1 > thresholds).float()  # [N,D]
        weights = weights.unsqueeze(2).repeat(1,1,4)

        # update mask based on NMS
        mask = self.nms_ers(teacher_output, teacher_scores, anchors, weights)


        return mask

    def ers_regression_2(self, teacher_output, teacher_scores, anchors):
        """
        :param: [N, D, 4, reg_max] tensor with reg outputs (probs) of teacher
        :param teacher_scores: [N, D, nc] tensor with cls output of teacher (after sigmoid)
        :param anchors: anchor points
        """

        # target_distri = target_distri.view(batch_size, num_preds, 4, reg_max)  # [N, D, 4, reg_max]
        #print(teacher_output.shape)
        alpha = 2
        D = teacher_output.shape[1]
        
        # top1 for each bbox
        top1 = torch.amax(teacher_output, dim=(-1,-2))  # [N, D]

        # get stats
        mean = top1.mean()
        std = top1.std()

        # derive threshold
        thresholds = mean + alpha * std
        #thresholds = thresholds.unsqueeze(-1).repeat(1,D)  # [N,D]

        # compute mask
        weights = (top1 > thresholds).float()  # [N,D]
        weights = weights.unsqueeze(-1).repeat(1,1,4)

        # update mask based on NMS
        #mask = self.nms_ers(teacher_output, teacher_scores, anchors, torch.ones_like(weights).float().to(teacher_output.device))
        #mask = (mask.bool() & weights.bool()).float()
        mask = self.nms_ers(teacher_output, teacher_scores, anchors, weights)
        

        return mask

    
    def nms_ers(self, target_distri, teacher_scores, anchors, mask):


        # get bbox from yolo output
        target_bboxes = self.bbox_decode(anchors, target_distri.detach())

        # cat bbox with cls output and prepare for NMS
        bboxes_pre_nms = torch.cat(( target_bboxes, teacher_scores), dim=-1)  # [N, D, 4+num_cls]
        bboxes_pre_nms = bboxes_pre_nms.permute(0,2,1)  # [N, 4+num_cls, D]

        # get list of masks using NMS (one per image)
        list_masks = self.non_max_suppression(bboxes_pre_nms, mask[:,:,0].bool())
        #list_masks = self.nms_boolean_mask(bboxes_pre_nms, mask[:,:,0].bool())

        # get one tensor and repeat for all offsets
        mask = torch.stack(list_masks, dim=0)

        mask = mask.unsqueeze(2).repeat(1,1,4)

        return mask
    
 

    def bbox_decode(self, anchor_points, pred_dist):
        """
        Decode predicted object bounding box coordinates from anchor points and
        distribution.
        """
        #print(pred_dist.shape)
        #b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = (
                pred_dist
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=True)



    def ers_cls(self, teacher_output):

        nc_classes = teacher_output.shape[-1]
        D = teacher_output.shape[1]
        alpha = 2
        target_scores = teacher_output  # [N, D, nc]

        top1 = torch.amax(target_scores, dim=-1)  # [N, D]
        mean = torch.mean(top1, dim=-1)
        std = torch.std(top1, dim=-1)  # [N]

        thresholds = mean + alpha * std
        thresholds = thresholds.unsqueeze(-1).repeat(1,D)  # [N,D]

        weights = (top1 > thresholds).float()  # [N,D]

        weights = weights.unsqueeze(2).repeat(1,1,nc_classes)

        #print(weights.sum())

        return weights
    
    def ers_cls_2(self, teacher_output):

        nc_classes = teacher_output.shape[-1]
        D = teacher_output.shape[1]
        alpha = 2
        target_scores = teacher_output  # [N, D, nc]

        top1 = target_scores
        mean = torch.mean(top1, dim=1)  # [N, nc]
        std = torch.std(top1, dim=1)  # [N, nc]


        thresholds = mean + alpha * std
        thresholds = thresholds.unsqueeze(1).repeat(1, D, 1)  # [N, D, nc]

        weights = (top1 > thresholds).float()  # [N, D, nc]


        return weights
    
    def ers_cls_3(self, teacher_output):


        nc_classes = teacher_output.shape[-1]
        D = teacher_output.shape[1]
        alpha = 2
        target_scores = teacher_output  # [N, D]

        top1 = torch.amax(target_scores, dim=-1)
  
        mean = top1.mean()
        std = top1.std()


        thresholds = mean + alpha * std
        #thresholds = thresholds.unsqueeze(1).repeat(1, D, 1)  # [N, D, nc]

        weights = (top1 > thresholds).float()  # [N, D]

        weights = weights.unsqueeze(-1).repeat(1,1,nc_classes)


        return weights
    
    def non_max_suppression(
            self,
            prediction,
            weights,
            conf_thres=0.05,
            iou_thres=0.005,
            classes=None,
            agnostic=False,
            multi_label=False,   # we set to false since here we need just the best class. consistent with original ERD
            labels=(),
            max_det=300,
            nc=0,  # number of classes (optional)
            max_nms=30000,
            max_wh=7680,
    ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Args:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        D = prediction.shape[-1]

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        nc_to_filter = len(self.classes)
        """consider just old classes"""
        xc = prediction[:, 4:(4+nc_to_filter)].amax(1) > conf_thres  # candidates  # [N, D]
        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = prediction.transpose(-1, -2)  # [N, D, 4+nc]

        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

        #output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        list_masks = []
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x [D, 24]
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            select = xc[xi] & weights[xi]  # get AND mask [D] tensor
            x = x[select]  # based on confidence, [P, 4+nc] tensor

            """not used"""
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
                v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)
            """not used until this"""

            # If none remain process next image
            if not x.shape[0]:
                list_masks.append(torch.zeros(D).float().to(device))  # mask all bboxes
                continue

            # Detections matrix nx6 (xyxy, conf, cls)  from [P, 4 + nc] to [P, 4], [P, nc], _, P= num pred filtered
            box, cls, mask = x.split((4, nc, nm), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                """again, consider just old classes"""
                conf, j = cls[:, :nc_to_filter].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            
            # x now is [P, 4+conf+cls], namely [P, 6]

            # Filter by class
            """not used"""
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                list_masks.append(torch.zeros(D))  # mask all bboxes
                continue

            """not used"""
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections


            # from indices to mask
            pre_mask = torch.zeros(boxes.shape[0]).float()
            pre_mask[i] = 1.0  # [P] vector

            # get the original shape: map mask of dim P to dim D
            final_mask = torch.zeros(D).float()
            final_mask[select] = pre_mask


            list_masks.append(deepcopy(final_mask).to(device))



        return list_masks
    


    def nms_boolean_mask(
        self,
        prediction,
        valid_mask,  # Boolean mask: shape (batch_size, num_boxes)
        conf_thres=0.05,
        iou_thres=0.6,
        max_det=300,
        max_nms=30000,
        max_wh=7680,
    ):


        device = prediction.device
        bs, num_boxes = prediction.shape[0], prediction.shape[2]  # batch size, number of boxes
        nc = prediction.shape[1] - 4  # number of classes
        mi = 4 + nc  # index where class scores start
        nc_to_filter = len(self.classes)

        # Convert [x, y, w, h] → [x1, y1, x2, y2]
        prediction = prediction.transpose(-1, -2)  # Shape: (batch, num_boxes, num_classes + 4)
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])

        output_masks = []
        for xi, x in enumerate(prediction):  # Process each batch separately
            valid = valid_mask[xi]  # Get the validity mask for this batch item
            conf_mask = x[:, 4:(4+nc_to_filter)].amax(1) > conf_thres  # Confidence threshold mask
            #print(valid[:,0].shape)
            #print(conf_mask.shape)
            #print(valid.shape)
            #print(conf_mask.shape)
            final_mask = valid & conf_mask  # Combine with validity mask

            x = x[final_mask]  # Apply the combined mask

            # Initialize a boolean mask for all boxes
            boolean_mask = torch.zeros(num_boxes, dtype=torch.bool, device=device)

            if not x.shape[0]:  # No valid boxes left
                output_masks.append(boolean_mask)
                continue

            # Extract boxes, scores, and classes
            box, cls = x[:, :4], x[:, 4:mi]
            conf, j = cls[:,:nc_to_filter].max(1, keepdim=True)  # Get best class confidence and index
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]  # Re-filter with confidence


            if not x.shape[0]:  # No boxes remain
                output_masks.append(boolean_mask)
                continue

            if x.shape[0] > max_nms:  # Limit number of boxes to process
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            # NMS
            c = x[:, 5:6] * max_wh  # Offset classes if needed
            boxes, scores = x[:, :4] + c, x[:, 4]  # Get boxes and scores

            keep_idx = torchvision.ops.nms(boxes, scores, iou_thres)[:max_det]  # Apply NMS


            # Update boolean mask (set True for kept indices)
            original_indices = final_mask.nonzero(as_tuple=True)[0]  # Map back to original indices
            boolean_mask[original_indices[keep_idx]] = True  # Mark selected boxes as True


            output_masks.append(boolean_mask.float())

        return output_masks  # List of [num_boxes] boolean masks per batch
