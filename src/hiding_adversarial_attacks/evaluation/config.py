from hiding_adversarial_attacks.visualization.config import (
    PCC_COLOR_PALETTE,
    SSIM_COLOR_PALETTE,
)

SIMILARITIES_FILE = "test_similarities.csv"
SIMILARITIES_COLS = ["pcc_sim", "mse_sim", "ssim_sim"]
Y_LABELS = ["PCC", "MSE", "SSIM"]
COLOR_PALETTES = [PCC_COLOR_PALETTE, "Oranges", SSIM_COLOR_PALETTE]
SIM_Y_LIMITS = [(-1.1, 1.1), None, (-0.1, 1.1)]
