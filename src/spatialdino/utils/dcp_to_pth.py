from pathlib import Path

from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

BASE_PATH = "/nfs/scratch2/shared_image_recog_ml/LiFT"
CKPT = Path(f"{BASE_PATH}/checkpoint_75.999.pth")

ckpt_path = (
    "/nfs/scratch2/shared_image_recog_ml/weights/dino3d_vits8_reg16/step=5999.ckpt"
)
ckpt_path_2 = (
    "/nfs/scratch2/shared_image_recog_ml/weights/dino3d_vits8_reg16/step=5999.ckpt"
)
TORCH_SAVE_CHECKPOINT_DIR2 = (
    "/nfs/scratch2/shared_image_recog_ml/weights/dino3d_vits8_reg16/step=5999.pth"
)

# convert dcp model to torch.save (assumes checkpoint was generated as above)
dcp_to_torch_save(ckpt_path_2, TORCH_SAVE_CHECKPOINT_DIR2)

# converts the torch.save model back to DCP
# dcp_to_torch_save(TORCH_SAVE_CHECKPOINT_DIR, f"{CHECKPOINT_DIR}_new")
