import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from pointbert.dvae import Group
from pointbert.dvae import Encoder
from pointbert.logger import print_log
import numpy as np
from pointbert.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder without hierarchical structure"""

    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, list)
                        else drop_path_rate
                    ),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


def patch_mix(lam, center):
    B, num_group, _ = center.size()
    b_group = int(num_group * lam)
    b_group = max(1, b_group)
    for b in range(B):
        i = torch.randperm(num_group)
        a_group_mask = torch.zeros([num_group - b_group])
        b_group_mask = torch.ones([b_group])
        mask = torch.cat((a_group_mask, b_group_mask), dim=0)[i].unsqueeze(0)
        if b == 0:
            random_group_mask = mask
        else:
            random_group_mask = torch.cat((random_group_mask, mask), dim=0)
    random_group_mask = random_group_mask.to(torch.bool)  # [B, G]
    return random_group_mask, b_group


class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        # self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        # self.encoder = Encoder(encoder_channel=self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        # self.cls_head_finetune = nn.Sequential(
        #     nn.Linear(self.trans_dim * 2, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, self.cls_dim)
        # )

        # self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith("transformer_q") and not k.startswith(
                "transformer_q.cls_head"
            ):
                base_ckpt[k[len("transformer_q.") :]] = base_ckpt[k]
            elif k.startswith("base_model"):
                base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log("missing_keys", logger="Transformer")
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger="Transformer",
            )
        if incompatible.unexpected_keys:
            print_log("unexpected_keys", logger="Transformer")
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger="Transformer",
            )

        print_log(
            f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
            logger="Transformer",
        )

    def perform_mix(self, pts):
        neighborhood, center = self.group_divider(pts)
        data = torch.cat([center.unsqueeze(2), neighborhood], dim=2)
        lam = np.random.random()
        points = neighborhood + center.unsqueeze(2)  # [B, G, M, 3]
        B, num_group, group_size, _ = points.size()
        # Determine paired two point clouds
        data = data.reshape(B // 2, 2, num_group, group_size + 1, -1)
        data_a = data[:, 0].clone()
        data_b = data[:, 1]
        data_c = data[:, 1].clone()

        center = center.reshape(B // 2, 2, num_group, -1)
        centers_a = center[:, 0]
        centers_b = center[:, 1]
        center = center.reshape(B, num_group, -1)

        points = points.reshape(B // 2, 2, num_group, group_size, -1)
        points_a = points[:, 0]
        points_b = points[:, 1]

        data_a, data_b, data_c = data_a.to("cuda"), data_b.to("cuda"), data_c.to("cuda")
        points_a, points_b = points_a.to("cuda"), points_b.to("cuda")

        group_mask, b_group = patch_mix(lam, center[: B // 2])
        fix_data_1 = torch.zeros_like(data_a)
        fix_data_2 = torch.zeros_like(data_c)

        fix_data_1 = data_a.clone()
        fix_data_1[group_mask] = data_c.clone()[group_mask]
        fix_data_2 = data_c.clone()
        fix_data_2[group_mask] = data_a.clone()[group_mask]
        data = torch.cat(
            [fix_data_1.unsqueeze(1), fix_data_2.unsqueeze(1)], dim=1
        ).reshape(B, num_group, group_size + 1, -1)

        return data[:, :, 1:, :], data[:, :, 0, :]

    def forward(self, pts, mix=False):
        # divide the point cloud in the same form. This is important
        if mix:
            neighborhood, center = self.perform_mix(pts)
        else:
            neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks

        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        return concat_f, x
