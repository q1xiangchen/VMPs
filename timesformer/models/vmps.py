import torch
import torch.nn as nn
import wandb

def rearrange_tensor(input_tensor, order):
    """
    Rearrange the input tensor to the desired order
    Args:
        input_tensor: input tensor to be rearranged 
        order: desired order of the tensor  (BTCHW)
    Returns:
        rearranged tensor
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    assert all([dim in "BCHWT" for dim in order]), "Order must not contain any characters other than BCHWT"

    return input_tensor.permute([order.index(dim) for dim in "BTCHW"])

def reverse_rearrange_tensor(input_tensor, order):
    """
    Rearrange the input tensor back to the original order
    Args:
        input_tensor: input tensor to be rearranged (BTCHW)
        order: original order of the tensor
    Returns:
        rearranged tensor
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    assert all([dim in "BCHWT" for dim in order]), "Order must not contain any characters other than BCHWT"

    return input_tensor.permute(["BTCHW".index(dim) for dim in order])

class VideoMotionPrompt(torch.nn.Module):
    def __init__(self, penalty_weight=0.0, wandb = False):
        super(VideoMotionPrompt, self).__init__()
        # default configs
        self.input_permutation = "BCTHW"   # Input permutation from video reader
        self.input_color_order = "BGR"     # Input color channel order from video reader
        self.gray_scale = {"B": 0.114, "G": 0.587, "R": 0.299}
        self.wandb = wandb

        # power normalization parameters
        self.pn = m_sigmoid
        self.m = nn.Parameter(nn.init.normal_(torch.zeros(1), 1e-5, 1))
        self.n = nn.Parameter(nn.init.normal_(torch.zeros(1), 1e-5, 1))

        # temporal attention variation regularization parameter
        self.lambda1 = penalty_weight
        
    def forward(self, video_seq):
        # rearrange the input tensor to BTCHW
        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        
        # normalize the input tensor back to [0, 1]
        norm_seq = video_seq * 0.225 + 0.45
        
        # transfor the input tensor to grayscale 
        weights = torch.tensor([self.gray_scale[idx] for idx in self.input_color_order], 
                               dtype=video_seq.dtype, device=video_seq.device)
        grayscale_video_seq = torch.einsum("btchw, c -> bthw", norm_seq, weights)

        ### frame difference ###
        B, T, H, W = grayscale_video_seq.shape
        frame_diff = grayscale_video_seq[:,1:] - grayscale_video_seq[:,:-1]

        ### power normalization ###
        attention_map = self.pn(frame_diff, self.m, self.n)
        repeat_attention_map = attention_map.unsqueeze(2).repeat(1, 1, 3, 1, 1)

        ### temporal attention variation regularization ###
        loss, temporal_loss = 0, 0
        if torch.is_grad_enabled():
            temp_diff = attention_map[:, 1:] - attention_map[:, :-1]
            temporal_loss = torch.sum(temp_diff.pow(2)) / (H*W*(T-2)*B)
            loss = self.lambda1 * temporal_loss

        ### log the loss and parameters ###
        if self.wandb:
            wandb.log({
                "m": self.m.data[0],
                "n": self.n.data[0],
                "temporal_loss": loss,
            })
        
        ### element-wise multiplication ###
        prompt = repeat_attention_map * video_seq[:,1:]
        motion_prompt = reverse_rearrange_tensor(prompt, self.input_permutation)

        return motion_prompt, loss


def m_sigmoid(input, m, n):
    return 1 / (1 + torch.exp(
        - (5 / (0.45 * torch.abs(torch.tanh(m)) + 1e-1)) * (input - 0.6 * torch.tanh(n))
        ))
