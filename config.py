from dataclasses import dataclass


@dataclass
class BackboneConfig:
    model_id:   str = "google/siglip-base-patch16-224"
    vision_dim: int = 768   # fixed for SigLIP Base
    lang_dim:   int = 768   # fixed for SigLIP Base
    image_size: int = 224   # SigLIP's trained resolution
    seq_len:    int = 64    # SigLIP tokenizer hard max


@dataclass
class RobotConfig:
    proprio_dim: int = 9    # e.g. 7-DOF arm positions + 2 gripper
    action_dim:  int = 7    # e.g. delta xyz + delta rpy + gripper


@dataclass
class PolicyConfig:
    t_embed_dim: int = 64
    hidden_dim:  int = 256
    num_layers:  int = 3
    num_inference_steps: int = 50


@dataclass
class TrainConfig:
    batch_size:  int   = 32
    lr:          float = 1e-4
    num_steps:   int   = 50_000
    val_every:   int   = 500
    data_root:   str   = "data/synthetic"
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    backbone: BackboneConfig = None
    robot:    RobotConfig    = None
    policy:   PolicyConfig   = None
    train:    TrainConfig    = None

    def __post_init__(self):
        if self.backbone is None: self.backbone = BackboneConfig()
        if self.robot    is None: self.robot    = RobotConfig()
        if self.policy   is None: self.policy   = PolicyConfig()
        if self.train    is None: self.train    = TrainConfig()

cfg = Config()
