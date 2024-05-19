"""
This script export pytorch_lightning module checkpoint,
which include many redundant information (e.g. optimizer,
scheduler, model state_dict, etc.), to pytorch model checkpoint
(only model state_dict & hyperparameters)

For detail, see ../neuralfp/module/audio_fingerprint
in function AudioFingerprint.export()
"""

import argparse

from neuralfp.module.audio_fingerprint import AudioFingerprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument(
        "-f",
        "--checkpoint",
        type=str,
        required=True,
        help="lightning module checkpoint path (.ckpt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="pytorch checkpoint save file name",
    )
    args = parser.parse_args()

    save_dir = args.output

    module = AudioFingerprint.load_from_checkpoint(args.checkpoint)
    module.export(filepath=save_dir)
    print(f"Successfully saved checkpoint to {save_dir}")
