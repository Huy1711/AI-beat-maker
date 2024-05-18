import torch
import torch.nn as nn
import torch.nn.functional as F


class NTxentLoss(nn.modules.loss._Loss):
    """
    Pytorch implementation of Normalized Temperature Crossentropy loss
    Original TensorFlow implementation from
        https://github.com/mimbres/neural-audio-fp/blob/main/model/fp/NTxent_loss_single_gpu.py
    """

    def __init__(self, tau=0.05):
        super().__init__()
        self.tau = tau

    def _drop_diag(self, x, n_anchors, mask_not_diag):
        x = torch.masked_select(x, mask_not_diag)
        return torch.reshape(x, (n_anchors, n_anchors - 1))

    def forward(
            self, 
            emb_org: torch.Tensor, 
            emb_rep: torch.Tensor,
            n_anchors: int
        ):
        labels = F.one_hot(
            torch.arange(0, n_anchors, device=emb_org.device),
            num_classes=n_anchors * 2 - 1,
        ).float()
        mask_not_diag = (1 - torch.eye(n_anchors, device=emb_org.device)) > 0

        logits_aa = torch.mm(emb_org, emb_org.t()) / self.tau
        logits_aa = self._drop_diag(logits_aa, n_anchors, mask_not_diag)

        logits_bb = torch.mm(emb_rep, emb_rep.t()) / self.tau
        logits_bb = self._drop_diag(logits_bb, n_anchors, mask_not_diag)

        logits_ab = torch.mm(emb_org, emb_rep.t()) / self.tau
        logits_ba = torch.mm(emb_rep, emb_org.t()) / self.tau

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss_a = cross_entropy_loss(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = cross_entropy_loss(torch.cat([logits_ba, logits_bb], dim=1), labels)
        return loss_a + loss_b
