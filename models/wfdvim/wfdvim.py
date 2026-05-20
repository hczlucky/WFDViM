import torch
import torch.nn as nn

from config.tiny_config_isic import setting_config
from models.wfdvim.network import WFDViMNet

class WFDViM(nn.Module):
    def __init__(
        self,
        input_channels=None,
        num_classes=None,
        embed_dims=None,
        layers=None,
        dims_decoder=None,
        depths_decoder=None,
        ssm_num=None,
        d_state=None,
        ssm_ratio=None,
        mlp_ratio=None,
        alphas_enc=None,
        alphas_dec=None,
        drop_path_rate=None,
        load_ckpt_path=None,
        freeze=False,
    ):
        super().__init__()
        model_config = setting_config.model_config
        input_channels = model_config["in_chans"] if input_channels is None else input_channels
        num_classes = model_config["num_classes"] if num_classes is None else num_classes
        embed_dims = model_config["embed_dims"] if embed_dims is None else embed_dims
        layers = model_config["layers"] if layers is None else layers
        dims_decoder = model_config["dims_decoder"] if dims_decoder is None else dims_decoder
        depths_decoder = model_config["depths_decoder"] if depths_decoder is None else depths_decoder
        ssm_num = model_config["ssm_num"] if ssm_num is None else ssm_num
        d_state = model_config["d_state"] if d_state is None else d_state
        ssm_ratio = model_config["ssm_ratio"] if ssm_ratio is None else ssm_ratio
        mlp_ratio = model_config["mlp_ratio"] if mlp_ratio is None else mlp_ratio
        alphas_enc = model_config["alphas_enc"] if alphas_enc is None else alphas_enc
        alphas_dec = model_config["alphas_dec"] if alphas_dec is None else alphas_dec
        drop_path_rate = model_config["drop_path_rate"] if drop_path_rate is None else drop_path_rate
        load_ckpt_path = model_config["load_ckpt_path"] if load_ckpt_path is None else load_ckpt_path

        self.num_classes = num_classes
        self.load_ckpt_path = load_ckpt_path
        self.model = WFDViMNet(
            in_chans=input_channels,
            num_classes=num_classes,
            embed_dims=embed_dims,
            layers=layers,
            dims_decoder=dims_decoder,
            depths_decoder=depths_decoder,
            ssm_num=ssm_num,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            mlp_ratio=mlp_ratio,
            alphas_enc=alphas_enc,
            alphas_dec=alphas_dec,
            drop_path_rate=drop_path_rate,
        )

        if self.load_ckpt_path:
            self._load_checkpoint(self.load_ckpt_path)

        if freeze:
            self._freeze_encoder()

    def _load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded from {ckpt_path}")
        print(f"Missing keys: {len(msg.missing_keys)}")
        print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    def _freeze_encoder(self):
        for param in self.model.stem.parameters():
            param.requires_grad = False
        for param in self.model.downsamples.parameters():
            param.requires_grad = False
        for param in self.model.encoder_stages.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)
