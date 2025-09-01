# exps/default/yolox_s_cow.py
from exps.default.yolox_s import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # ----- dataset -----
        self.num_classes = 1
        self.data_dir = "sample_training"  # has train/, valid/, annotations/
        self.train_ann = "instances_train.json"  # in data_dir/annotations/
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"  # optional

        # folders that contain images
        self.train_name = "train"
        self.val_name = "valid"
        self.test_name = "test"

        # ----- schedule / aug (tweak as needed) -----
        self.max_epoch = 50
        self.eval_interval = 5
        # For small datasets, you can soften aug:
        # self.mosaic_prob = 0.0
        # self.mixup_prob  = 0.0
