from datetime import datetime

from datasets.dataset import SegmentationTransform
from utils import BceDiceLoss, DeepSupervisionLoss

class setting_config:
    network = "WFDViM"
    datasets = "isic"
    data_path = "/path/to/ISIC/"

    model_config = {
        "num_classes": 1,
        "in_chans": 3,
        "embed_dims": [48, 96, 192, 384],
        "layers": [4, 3, 10, 5],
        "dims_decoder": [384, 192, 96, 48],
        "depths_decoder": [5, 10, 3, 4],
        "ssm_num": 1,
        "d_state": 1,
        "ssm_ratio": 2.0,
        "mlp_ratio": 4.0,
        "alphas_enc": [0.25, 0.5, 0.5, 0.75],
        "alphas_dec": [0.75, 0.5, 0.5, 0.25],
        "drop_path_rate": 0.2,
        "ssm_drop_rate": 0.0,
        "mlp_drop_rate": 0.0,
        "load_ckpt_path": None,
    }

    base_criterion = BceDiceLoss(wd=1, wb=1)
    criterion = DeepSupervisionLoss(base_loss=base_criterion)

    pretrained_path = None
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    train_transformer = SegmentationTransform(img_size=input_size_h, train=True)
    test_transformer = SegmentationTransform(img_size=input_size_h, train=False)
    distributed = False
    local_rank = -1
    num_workers = 8
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = "0"
    batch_size = 32
    epochs = 300

    work_dir = "results/" + network + "_" + datasets + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "/"

    print_interval = 20
    val_interval = 1
    save_interval = 100
    threshold = 0.5
    only_test_and_save_figs = False
    best_ckpt_path = None
    img_save_path = None

    opt = "AdamW"
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False

    sch = "CosineAnnealingLR"
    T_max = epochs
    eta_min = 1e-5
    last_epoch = -1
