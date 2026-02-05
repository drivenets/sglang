from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputChecker,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputChecker,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPConfig,
    DeepEPDispatcher,
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
)
try:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferDispatcher,
        FlashinferDispatchOutput,
    )
except (ImportError, AssertionError):
    # flashinfer not available on ROCm
    FlashinferDispatcher = None
    FlashinferDispatchOutput = None
from sglang.srt.layers.moe.token_dispatcher.fuseep import NpuFuseEPDispatcher
from sglang.srt.layers.moe.token_dispatcher.mooncake import (
    MooncakeCombineInput,
    MooncakeDispatchOutput,
    MooncakeEPDispatcher,
)
try:
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPDispatcher,
        MoriEPNormalCombineInput,
        MoriEPNormalDispatchOutput,
    )
except (ImportError, AssertionError):
    # moriep may have flashinfer dependencies
    MoriEPDispatcher = None
    MoriEPNormalCombineInput = None
    MoriEPNormalDispatchOutput = None
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatcher,
    StandardDispatchOutput,
)

__all__ = [
    "BaseDispatcher",
    "BaseDispatcherConfig",
    "CombineInput",
    "CombineInputChecker",
    "CombineInputFormat",
    "DispatchOutput",
    "DispatchOutputFormat",
    "DispatchOutputChecker",
    "FlashinferDispatchOutput",
    "FlashinferDispatcher",
    "MooncakeCombineInput",
    "MooncakeDispatchOutput",
    "MooncakeEPDispatcher",
    "MoriEPNormalDispatchOutput",
    "MoriEPNormalCombineInput",
    "MoriEPDispatcher",
    "StandardDispatcher",
    "StandardDispatchOutput",
    "StandardCombineInput",
    "DeepEPConfig",
    "DeepEPDispatcher",
    "DeepEPNormalDispatchOutput",
    "DeepEPLLDispatchOutput",
    "DeepEPLLCombineInput",
    "DeepEPNormalCombineInput",
    "NpuFuseEPDispatcher",
]
