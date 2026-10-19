import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class DetectionLoss(nn.Module):
    def __init__(
        self,
        grid_size=32,
        boxes_per_cell=3,
        lambda_box=5.0,
        lambda_obj=1.0,
        lambda_noobj=10.0,
        lambda_cls=1.0,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell

        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

    @staticmethod
    def xywh_to_xyxy(boxes):
        """Convert (..., cx, cy, w, h) to (..., x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(dim=-1)

        return torch.stack(
            (
                cx - w / 2,
                cy - h / 2,
                cx + w / 2,
                cy + h / 2,
            ),
            dim=-1,
        )

    @staticmethod
    def pairwise_iou(boxes1, boxes2, eps=1e-7):
        """Pairwise IoU for N and M boxes in xyxy format."""
        top_left = torch.maximum(
            boxes1[:, None, :2],
            boxes2[None, :, :2],
        )
        bottom_right = torch.minimum(
            boxes1[:, None, 2:],
            boxes2[None, :, 2:],
        )

        intersection_wh = (
            bottom_right - top_left
        ).clamp(min=0)

        intersection = (
            intersection_wh[..., 0] * intersection_wh[..., 1]
        )

        area1 = (
            (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
            * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        )

        area2 = (
            (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
            * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        )

        union = area1[:, None] + area2[None, :] - intersection

        return intersection / union.clamp(min=eps)

    @staticmethod
    def giou_loss(pred_boxes, true_boxes, eps=1e-7):
        """GIoU loss for corresponding xyxy boxes."""
        top_left = torch.maximum(
            pred_boxes[:, :2],
            true_boxes[:, :2],
        )
        bottom_right = torch.minimum(
            pred_boxes[:, 2:],
            true_boxes[:, 2:],
        )

        intersection_wh = (
            bottom_right - top_left
        ).clamp(min=0)

        intersection = (
            intersection_wh[:, 0] * intersection_wh[:, 1]
        )

        pred_wh = (
            pred_boxes[:, 2:] - pred_boxes[:, :2]
        ).clamp(min=0)

        true_wh = (
            true_boxes[:, 2:] - true_boxes[:, :2]
        ).clamp(min=0)

        area_pred = pred_wh[:, 0] * pred_wh[:, 1]
        area_true = true_wh[:, 0] * true_wh[:, 1]

        union = area_pred + area_true - intersection
        iou = intersection / union.clamp(min=eps)

        enclosing_tl = torch.minimum(
            pred_boxes[:, :2],
            true_boxes[:, :2],
        )
        enclosing_br = torch.maximum(
            pred_boxes[:, 2:],
            true_boxes[:, 2:],
        )

        enclosing_wh = (
            enclosing_br - enclosing_tl
        ).clamp(min=0)

        enclosing_area = (
            enclosing_wh[:, 0] * enclosing_wh[:, 1]
        )

        giou = iou - (
            (enclosing_area - union).clamp(min=0)
            / enclosing_area.clamp(min=eps)
        )

        return 1.0 - giou

    def forward(self, predictions, targets):
        # Both: [B, 32, 32, 3, 8]
        expected_shape = (
            self.grid_size,
            self.grid_size,
            self.boxes_per_cell,
            8,
        )

        if predictions.ndim != 5:
            raise ValueError(
                f"Expected 5D predictions, got {predictions.shape}"
            )

        if tuple(predictions.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected predictions ending in {expected_shape}, "
                f"got {predictions.shape}"
            )

        if targets.shape != predictions.shape:
            raise ValueError(
                f"Target shape {targets.shape} does not match "
                f"prediction shape {predictions.shape}"
            )

        batch_size = predictions.shape[0]
        grid = self.grid_size
        slots = self.boxes_per_cell
        device = predictions.device

        # Decode each predicted box to normalized image coordinates.
        rows = torch.arange(grid, device=device).view(1, grid, 1)
        cols = torch.arange(grid, device=device).view(1, 1, grid)

        raw_boxes = predictions[..., :4].sigmoid()

        pred_cx = (
            cols.unsqueeze(-1) + raw_boxes[..., 0]
        ) / grid

        pred_cy = (
            rows.unsqueeze(-1) + raw_boxes[..., 1]
        ) / grid

        pred_wh = raw_boxes[..., 2:4]

        decoded_pred = torch.cat(
            (
                pred_cx.unsqueeze(-1),
                pred_cy.unsqueeze(-1),
                pred_wh,
            ),
            dim=-1,
        )

        pred_xyxy = self.xywh_to_xyxy(decoded_pred)

        # Decode ground-truth centres from cell-relative to image-relative.
        true_cx = (
            cols.unsqueeze(-1) + targets[..., 0]
        ) / grid

        true_cy = (
            rows.unsqueeze(-1) + targets[..., 1]
        ) / grid

        decoded_true = torch.cat(
            (
                true_cx.unsqueeze(-1),
                true_cy.unsqueeze(-1),
                targets[..., 2:4],
            ),
            dim=-1,
        )

        true_xyxy = self.xywh_to_xyxy(decoded_true)

        gt_mask = targets[..., 4] > 0.5

        # Each target is matched to a unique prediction in its cell.
        positive_mask = torch.zeros_like(gt_mask)
        box_losses = []
        class_losses = []

        for b in range(batch_size):
            occupied_cells = gt_mask[b].any(dim=-1).nonzero(
                as_tuple=False
            )

            for cell in occupied_cells:
                row = int(cell[0].item())
                col = int(cell[1].item())

                gt_slots = gt_mask[b, row, col].nonzero(
                    as_tuple=False
                ).flatten()

                pred_boxes = pred_xyxy[b, row, col]
                gt_boxes = true_xyxy[b, row, col, gt_slots]

                ious = self.pairwise_iou(
                    pred_boxes.detach(),
                    gt_boxes.detach(),
                )

                num_gt = len(gt_slots)

                best_assignment = None
                best_total_iou = float("-inf")

                for pred_perm in permutations(range(slots), num_gt):
                    total_iou = sum(
                        ious[pred_perm[gt_idx], gt_idx].item()
                        for gt_idx in range(num_gt)
                    )

                    if total_iou > best_total_iou:
                        best_total_iou = total_iou
                        best_assignment = pred_perm

                assignments = [
                    (pred_perm, gt_slots[gt_idx])
                    for gt_idx, pred_perm in enumerate(best_assignment)
                ]

                for pred_slot, gt_slot in assignments:
                    positive_mask[b, row, col, pred_slot] = True

                    box_losses.append(
                        self.giou_loss(
                            pred_xyxy[
                                b, row, col, pred_slot
                            ].unsqueeze(0),
                            true_xyxy[
                                b, row, col, gt_slot
                            ].unsqueeze(0),
                        )[0]
                    )

                    class_id = targets[
                        b, row, col, gt_slot, 5:
                    ].argmax()

                    class_losses.append(
                        F.cross_entropy(
                            predictions[
                                b, row, col, pred_slot, 5:
                            ].unsqueeze(0),
                            class_id.unsqueeze(0),
                        )
                    )

        if box_losses:
            box_loss = torch.stack(box_losses).mean()
            cls_loss = torch.stack(class_losses).mean()
        else:
            box_loss = predictions[..., :4].sum() * 0
            cls_loss = predictions[..., 5:].sum() * 0

        # Objectness loss with properly balanced background suppression
        obj_logits = predictions[..., 4]

        pos_obj_mask = positive_mask
        neg_obj_mask = ~positive_mask

        if pos_obj_mask.any():
            obj_pos_loss = F.binary_cross_entropy_with_logits(
                obj_logits[pos_obj_mask],
                torch.ones_like(obj_logits[pos_obj_mask]),
            )
        else:
            obj_pos_loss = obj_logits.sum() * 0

        if neg_obj_mask.any():
            obj_neg_loss = F.binary_cross_entropy_with_logits(
                obj_logits[neg_obj_mask],
                torch.zeros_like(obj_logits[neg_obj_mask]),
            )
        else:
            obj_neg_loss = obj_logits.sum() * 0

        total_loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * obj_pos_loss
            + self.lambda_noobj * obj_neg_loss
            + self.lambda_cls * cls_loss
        )

        components = {
            "box_giou": box_loss.detach(),
            "objectness_positive": obj_pos_loss.detach(),
            "objectness_negative": obj_neg_loss.detach(),
            "classification": cls_loss.detach(),
        }

        return total_loss, components