import os

print("INFO: Init script")

os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/notebooks/repos/EWP/SIE_3DIE/torch_compile"
os.environ["TRITON_CACHE_DIR"] = "/notebooks/repos/EWP/SIE_3DIE/torch_compile/triton"
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"

import torch

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

dynamo_config = torch._dynamo.config
dynamo_config.compiled_autograd = True
dynamo_config.capture_scalar_outputs = False
dynamo_config.cache_size_limit = 512
