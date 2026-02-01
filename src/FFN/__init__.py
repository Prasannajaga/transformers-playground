from .RELU import RELU_FFN
from .GELU import GELU_FFN
from .SILU import SILU_FFN
from .SWIGLU import SWIGLU_FFN

FFN_REGISTRY = {
    "relu": RELU_FFN,
    "gelu": GELU_FFN,
    "silu": SILU_FFN, 
    "swiglu": SWIGLU_FFN, 
}
