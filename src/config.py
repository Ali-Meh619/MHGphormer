def dBm_watt(x):
    return 10**(x/10)/1000

# Using default values as parsed from the original code
import torch

ARGS = {
    "IRS_elements": 64, "IRS_elements_row": 8, "BS_row": 4, "BS_col": 8, "BS_antenna": 32, "num_users": 6,
    "P_max": dBm_watt(30), "noise_pow": dBm_watt(-174),
    "loc_BS": torch.tensor((25, -20, -5)), "loc_IRS": torch.tensor((0, 0, 0)), "user_range_x1": (0, 15),
    "user_range_x2": (-2.5, -10), "user_range_y": (0, 25), "node_types": 3, "sub_bands": 5, "f_start": 0.380e12,
    "f_end": 0.4e12, "r_u_min": 13e9, "user_antenna": 2, "hidden": 1024, "dropout": 0.005, "eta_1": 111.48,
    # NOTE: "int_samp": 100000 with "batch": 250 requires >80GB of GPU VRAM. 
    # If you encounter CUDA Out of Memory errors, reduce `int_samp` (e.g. to 10000) or reduce `batch`.
    "eta_2": -2.97e-10, "eta_3": 0.01, "int_samp": 100000, "lam1": 11e9, "lam2": 11e9, "user_range_z": -10,
    "Rician_factor": 1, "b_g": 0.75e9, "b_max": 4e9, "antenna_space": 395e-6, "IRS_space": 395e-6,
    "inp_dim": 2, "hid_dim": 4, "num_metapath": 5, "train_s": 12000, "validation_s": 260, "test_s": 2700,
    "samples": 14960, "batch": 250, "lr_init": 0.0005, "epochs": 250, "log_step": 25, "lr_decay": True
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
