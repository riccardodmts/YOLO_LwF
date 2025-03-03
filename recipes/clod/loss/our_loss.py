import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors


class OurLoss(v8DetectionLoss):
    def __init__(self, h, m, device, old_classes, consts, new_classes=[]):  # model must be de-paralleled
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.consts = consts
        self.old_classes = old_classes
        self.new_classes = new_classes
        self.classes = old_classes + new_classes

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # LwF on the output
        self.distill_loss = torch.nn.MSELoss()
        self.backbone_loss = consts[0]
        self.neck_loss = consts[1]

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

    def __call__(self, preds, batch, student_inter_feats, teacher_inter_feats):
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
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
        # Cls loss
        """
        if len(self.classes):
            loss[1] = (
                self.bce(pred_scores[:,:,self.classes], target_scores[:,:,self.classes].to(dtype)).sum() / target_scores_sum
            )  # BCE
        else:
            loss[1] = (
               self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum 
            )
        """

        pred_scores[batch_size//2:, :, self.new_classes] = -100000

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

        backbone_loss = 0
        neck_loss = 0
        distill_loss = 0
        filter_idx = self.reg_max * 4 + self.new_classes[0]

        if self.consts[0] > 0:
            for i in range(3):
                backbone_loss += self.distill_loss(student_inter_feats[i][:, : filter_idx,:,:],
                                                teacher_inter_feats[i][:, :filter_idx,:,:].detach())
                
            distill_loss += self.consts[0]*backbone_loss
                
            if self.consts[1] > 0:

                for i in range(3):
                    neck_loss += self.distill_loss(student_inter_feats[i+3][:, : filter_idx,:,:],
                                        teacher_inter_feats[i+3][:, :filter_idx,:,:].detach())
                    
                distill_loss += neck_loss * self.consts[1]

        else:
            for i in range(3):
                    neck_loss += self.distill_loss(student_inter_feats[i][:, : filter_idx,:,:],
                                        teacher_inter_feats[i][:, :filter_idx,:,:].detach())
                    
            distill_loss += neck_loss * self.consts[1]

        distill_loss /= len(teacher_inter_feats)

        total_loss = loss.sum() * batch_size + distill_loss * batch_size

        return total_loss, loss.detach()