import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.metrics import bbox_iou
from loss.LwFloss import YOLOv8LwFLossNew, LwFLoss
from loss.erd_loss import ERS

class YOLOv8LwFLoss(nn.Module):

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
        IoU_scores = torch.pow(self.score_IoU(student_reg_output, target_distri, anchors).repeat(1,1,len(self.classes)), 6)

        target_distri = target_distri.view(batch_size, num_preds, 4, reg_max)  # [N, D, 4, reg_max]
        target_scores = self.sigmoid(target_logit_scores)  # [N, D, nc]



        # reshape regression output for student
        pred_distri = student_reg_output.view(batch_size, num_preds, 4, reg_max)  # [N, D, 4, reg_max]

        """classification"""
        lwf_cls_loss = IoU_scores * self.bce(student_cl_output[:,:, self.classes], target_scores[:,:, self.classes].detach())

        lwf_cls_loss = torch.mean(lwf_cls_loss)

        """regression"""
        # compute weights
        weights, _ = torch.max(target_scores[:, :, self.classes], dim=2)  # [N, D]
        # repeat
        weights = weights.unsqueeze(2).repeat(1,1,4)  # [N, D, 4]

        # compute CEs
        target_distri = self.softmax(target_distri.detach())  # p
        log_pred_distri = self.log_softmax(pred_distri)  # log q
        CEs = torch.sum( - target_distri * log_pred_distri, dim=3)  # [N, D, 4]

        # weight CEs
        weighted_CEs = torch.pow(weights, 1) * CEs

        lwf_regression_loss = torch.mean(weighted_CEs)

        lwf_loss = self.c1 * lwf_cls_loss + self.c2 * lwf_regression_loss

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
    





"""YOLOv8 loss for YOLO LwF + OCDM"""
class LwFLossReplay2(v8DetectionLoss):
    def __init__(self, h, m, device, c1=1.0, c2=1.0, c3=None, old_classes=[], classes=None, cfg=[True, False], mask_labels=None):  # model must be de-paralleled

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
        self.use_new_images = cfg[0]
        self.use_labels = cfg[1]
        self.old_classes = old_classes

        self.classes_per_task = mask_labels

    def _create_mask(self, task_ids, num_preds):

        mask = torch.ones((len(task_ids), num_preds, self.nc), dtype=torch.float32)
        all_class_ids = range(self.nc)

        for j, task_id in enumerate(task_ids):
            mask[j,:,[i for i in all_class_ids if i not in self.classes_per_task[task_id]]] = 0.0

        return mask


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
        num_labels = batch["num_labels"]
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        teacher_output = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        batch_size = pred_scores.shape[0]

        # get output student for lwf. if current task images are used, take all of them otherwise half batch
        pred_scores_lwf = pred_scores if self.use_new_images else pred_scores[batch_size//2:]
        pred_distri_lwf = pred_distri if self.use_new_images else pred_distri[batch_size//2:]

        # get output teacher for lwf. if current task images are used, take all of them otherwise half batch
        if not self.use_new_images:
            if isinstance(teacher_output, list):
                teacher_output_lwf = [l[batch_size//2:] for l in teacher_output]
            else:
                teacher_output_lwf = teacher_output[batch_size//2:]
        else:
            teacher_output_lwf = teacher_output

        task_id = batch.pop("task_id", None)  # used for masking

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        """LWF LOSS"""
        lwf_loss = self.lwf_loss(pred_scores_lwf, pred_distri_lwf, teacher_output_lwf, anchor_points)


        """LOSS WITH LABELS"""

        dtype = pred_scores.dtype
        pred_scores = pred_scores[:batch_size//2]
        pred_distri = pred_distri[:batch_size//2]
        
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)

        

        # Targets: if labels from replay memeory used, 
        if self.use_labels:


            #targets for current task
            targets = torch.cat(
                (batch["batch_idx"][:num_labels].view(-1, 1), batch["cls"][:num_labels].view(-1, 1), batch["bboxes"][:num_labels]),
                1,
            )

            #targets for previous tasks
            targets_old = torch.cat(
                ((batch["batch_idx"][num_labels:].view(-1, 1)-batch_size//2), batch["cls"][num_labels:].view(-1, 1), batch["bboxes"][num_labels:]),
                1,
            )

            targets_old = self.preprocess(
                targets_old.to(self.device), batch_size//2, scale_tensor=imgsz[[1, 0, 1, 0]]
            )

            # compute loss for replay samples (task ids are passed to mask properly classes)
            replay_loss = self.loss_replay(pred_distri_lwf[batch_size//2:], pred_scores_lwf[batch_size//2:], targets_old, anchor_points, stride_tensor, task_id)
        else:
            targets = torch.cat(
                (batch["batch_idx"][:num_labels].view(-1, 1), batch["cls"][:num_labels].view(-1, 1), batch["bboxes"][:num_labels]),
                1,
            )


        targets = self.preprocess(
            targets.to(self.device), batch_size//2, scale_tensor=imgsz[[1, 0, 1, 0]]
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
            # mask old classes for samples from current task
            loss[1] = (
                self.bce(pred_scores[:,:,self.classes], target_scores[:,:,self.classes].to(dtype)).sum() / target_scores_sum
            )  # BCE
        else:
            loss[1] = (
               self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum 
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


        self.last_yolo_loss = loss.sum().item()
        self.last_lwf_loss = lwf_loss.item()

        batch_size_lwf = batch_size if self.use_new_images else batch_size//2
        batch_size = batch_size//2

        """Sum loss for reducing forgetting"""
        cl_loss = lwf_loss * batch_size_lwf
        if self.use_labels:
            cl_loss += replay_loss
        
        """Compute total loss"""
        lwf_gain = 1.0
        total_loss = loss.sum() * batch_size * (2.0-lwf_gain) + lwf_gain * cl_loss

        return total_loss, loss.detach()
    
    def loss_replay(self, pred_distri, pred_scores, targets, anchor_points, stride_tensor, task_id=None):


        task_id = task_id[len(task_id)//2:] if task_id is not None else None
        dtype = pred_scores.dtype
        batch_size = pred_distri.shape[0]

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) 

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
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        if task_id is not None:

            mask = self._create_mask(task_id, pred_distri.shape[1]).to(self.device)
            loss[1] = (
                (self.bce(pred_scores, target_scores.to(dtype)) * mask).sum() / target_scores_sum
            )           
        else:

            loss[1] = (
                self.bce(pred_scores[:,:,self.old_classes], target_scores[:,:,self.old_classes].to(dtype)).sum() / target_scores_sum
            ) 


        #mask = self._create_mask(task_id, pred_distri.shape[1]).to(self.device)
        """
        print(mask[:,0,:])
        loss[1] = (
                self.bce(pred_scores[:,:,self.old_classes], target_scores[:,:,self.old_classes].to(dtype)).sum() / target_scores_sum
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


        return loss.sum() * batch_size
    

"""YOLOv8 loss for LwF + OCDM"""
class NaiveLwFLossReplay(v8DetectionLoss):
    def __init__(self, h, m, device, lwf=(3.0, 1.0), new_classes=[]):  # model must be de-paralleled
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
        self.lwf_loss = torch.nn.MSELoss(reduction="sum")
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

    def __call__(self, preds, batch, teacher_output, optional = None):
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        """

        num_labels = batch["num_labels"]

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


        targets = torch.cat(
                (batch["batch_idx"][:num_labels].view(-1, 1), batch["cls"][:num_labels].view(-1, 1), batch["bboxes"][:num_labels]),
                1,
            )
        
        # consider just half batch for labels (the samples from current task)
        pred_scores = pred_scores[:batch_size//2]
        pred_distri = pred_distri[:batch_size//2]

        targets = self.preprocess(
            targets.to(self.device), batch_size//2, scale_tensor=imgsz[[1, 0, 1, 0]]
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
        lwf_loss_temp = [0,0,0]
        for i in range(3):
            
            #lwf_loss += self.lwf_loss(feats[i][:, : filter_idx,:,:], teacher_output[i][:, :filter_idx,:,:].detach())
            
            lwf_loss_temp[i] += self.lwf[0] * self.lwf_loss(feats[i][:,(self.reg_max * 4) : filter_idx,:,:], teacher_output[i][:,(self.reg_max * 4) :filter_idx,:,:].detach())
            lwf_loss_temp[i] += self.lwf[1]*self.lwf_loss(feats[i][:,:(self.reg_max * 4),:,:], teacher_output[i][:, :(self.reg_max * 4),:,:].detach())
            a,b,c,d = feats[i].shape
            lwf_loss_temp[i] /= (a*b*c*d)
            lwf_loss += lwf_loss_temp[i]
            

        lwf_loss /= 3

        total_loss = loss.sum() * batch_size +  lwf_loss * batch_size

        return total_loss, loss.detach()
    



class LwFLossReplayERD(v8DetectionLoss):
    def __init__(self, h, m, device, c1=1.0, c2=1.0, c3=None, old_classes=[], classes=None, cfg=[True, False], mask_labels=None):  # model must be de-paralleled

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

        self.lwf_loss = ERS(c1, c2, old_classes, m.reg_max, device, c3)
        self.last_yolo_loss = 0
        self.last_lwf_loss = 0
        self.use_new_images = cfg[0]
        self.use_labels = cfg[1]
        self.old_classes = old_classes

        self.classes_per_task = mask_labels

    def _create_mask(self, task_ids, num_preds):

        mask = torch.ones((len(task_ids), num_preds, self.nc), dtype=torch.float32)
        all_class_ids = range(self.nc)

        for j, task_id in enumerate(task_ids):
            mask[j,:,[i for i in all_class_ids if i not in self.classes_per_task[task_id]]] = 0.0

        return mask


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
        num_labels = batch["num_labels"]
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        teacher_output = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        batch_size = pred_scores.shape[0]

        # get output student for lwf. if current task images are used, take all of them otherwise half batch
        pred_scores_lwf = pred_scores if self.use_new_images else pred_scores[batch_size//2:]
        pred_distri_lwf = pred_distri if self.use_new_images else pred_distri[batch_size//2:]

        # get output teacher for lwf. if current task images are used, take all of them otherwise half batch
        if not self.use_new_images:
            if isinstance(teacher_output, list):
                teacher_output_lwf = [l[batch_size//2:] for l in teacher_output]
            else:
                teacher_output_lwf = teacher_output[batch_size//2:]
        else:
            teacher_output_lwf = teacher_output

        task_id = batch.pop("task_id", None)  # used for masking

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        """LWF LOSS"""
        lwf_loss = self.lwf_loss(pred_scores_lwf, pred_distri_lwf, teacher_output_lwf, anchor_points)


        """LOSS WITH LABELS"""

        dtype = pred_scores.dtype
        pred_scores = pred_scores[:batch_size//2]
        pred_distri = pred_distri[:batch_size//2]
        
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)

        

        # Targets: if labels from replay memeory used, 
        if self.use_labels:


            #targets for current task
            targets = torch.cat(
                (batch["batch_idx"][:num_labels].view(-1, 1), batch["cls"][:num_labels].view(-1, 1), batch["bboxes"][:num_labels]),
                1,
            )

            #targets for previous tasks
            targets_old = torch.cat(
                ((batch["batch_idx"][num_labels:].view(-1, 1)-batch_size//2), batch["cls"][num_labels:].view(-1, 1), batch["bboxes"][num_labels:]),
                1,
            )

            targets_old = self.preprocess(
                targets_old.to(self.device), batch_size//2, scale_tensor=imgsz[[1, 0, 1, 0]]
            )

            # compute loss for replay samples (task ids are passed to mask properly classes)
            replay_loss = self.loss_replay(pred_distri_lwf[batch_size//2:], pred_scores_lwf[batch_size//2:], targets_old, anchor_points, stride_tensor, task_id)
        else:
            targets = torch.cat(
                (batch["batch_idx"][:num_labels].view(-1, 1), batch["cls"][:num_labels].view(-1, 1), batch["bboxes"][:num_labels]),
                1,
            )


        targets = self.preprocess(
            targets.to(self.device), batch_size//2, scale_tensor=imgsz[[1, 0, 1, 0]]
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
            # mask old classes for samples from current task
            loss[1] = (
                self.bce(pred_scores[:,:,self.classes], target_scores[:,:,self.classes].to(dtype)).sum() / target_scores_sum
            )  # BCE
        else:
            loss[1] = (
               self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum 
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


        self.last_yolo_loss = loss.sum().item()
        self.last_lwf_loss = lwf_loss.item()

        batch_size_lwf = batch_size if self.use_new_images else batch_size//2
        batch_size = batch_size//2

        """Sum loss for reducing forgetting"""
        cl_loss = lwf_loss * batch_size_lwf
        if self.use_labels:
            cl_loss += replay_loss
        
        """Compute total loss"""
        lwf_gain = 1.0
        total_loss = loss.sum() * batch_size * (2.0-lwf_gain) + lwf_gain * cl_loss

        return total_loss, loss.detach()
    
    def loss_replay(self, pred_distri, pred_scores, targets, anchor_points, stride_tensor, task_id=None):


        task_id = task_id[len(task_id)//2:] if task_id is not None else None
        dtype = pred_scores.dtype
        batch_size = pred_distri.shape[0]

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) 

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
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        if task_id is not None:

            mask = self._create_mask(task_id, pred_distri.shape[1]).to(self.device)
            loss[1] = (
                (self.bce(pred_scores, target_scores.to(dtype)) * mask).sum() / target_scores_sum
            )           
        else:

            loss[1] = (
                self.bce(pred_scores[:,:,self.old_classes], target_scores[:,:,self.old_classes].to(dtype)).sum() / target_scores_sum
            ) 


        #mask = self._create_mask(task_id, pred_distri.shape[1]).to(self.device)
        """
        print(mask[:,0,:])
        loss[1] = (
                self.bce(pred_scores[:,:,self.old_classes], target_scores[:,:,self.old_classes].to(dtype)).sum() / target_scores_sum
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


        return loss.sum() * batch_size