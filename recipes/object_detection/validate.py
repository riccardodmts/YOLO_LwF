"""
YOLO training.

This code allows you to validate an object detection model.

To run this script, you can start it with:
    python validate.py cfg/<cfg_file>.py <checkpoint_path>

Authors:
    - Matteo Beltrami, 2024
    - Francesco Paissan, 2024
"""

from micromind.utils import parse_configuration
from micromind.utils.yolo import (
    load_config,
)
import sys
from validation.validator import DetectionValidator

from train import YOLO, replace_datafolder


class YOLO(YOLO):
    def __init__(self, m_cfg, hparams, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(m_cfg, hparams, *args, **kwargs)
        self.m_cfg = m_cfg
        self.device = "cuda"

    def forward(self, img):
        """Runs the forward method by calling every module."""
        backbone = self.modules["backbone"](img)
        if "sppf" in self.modules.keys():
            neck_input = backbone[1]
            neck_input.append(self.modules["sppf"](backbone[0]))
        else:
            neck_input = backbone
        neck = self.modules["neck"](*neck_input)
        head = self.modules["head"](neck)

        return head


if __name__ == "__main__":
    assertion_msg = "Usage: python validate.py <hparams_config> <model_weights>"
    assert len(sys.argv) >= 3, assertion_msg

    hparams = parse_configuration(sys.argv[1])
    m_cfg, data_cfg = load_config(hparams.data_cfg)
    data_cfg = replace_datafolder(hparams, data_cfg)

    m_cfg.imgsz = hparams.input_shape[-1]  # temp solution

    model_weights_path = sys.argv[2]
    args = dict(model="yolov8n.pt", data=hparams.data_cfg, verbose=False, plots=False)
    validator = DetectionValidator(args=args)

    model = YOLO(m_cfg, hparams)
    model.load_modules(model_weights_path)

    val = validator(model=model)

    print("METRICS:")
    print("Box map50-95:", round(validator.metrics.box.map * 100, 3), "%")
    print("Box map50:", round(validator.metrics.box.map50 * 100, 3), "%")
    print("Box map75:", round(validator.metrics.box.map75 * 100, 3), "%")
