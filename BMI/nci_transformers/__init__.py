# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "3.4.0_bmi"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

# Integrations: this needs to come before other ml imports
# in order to allow any 3rd-party code to initialize properly
from .integrations import (  # isort:skip
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)

# Configurations
from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from .configuration_utils import PretrainedConfig

# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_datasets_available,
    is_faiss_available,
    is_flax_available,
    is_psutil_available,
    is_py3nvml_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_tpu_available,
)
from .hf_argparser import HfArgumentParser


# Tokenizers
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TensorType,
    TokenSpan,
)

if is_sentencepiece_available():
    from .tokenization_t5 import T5Tokenizer
else:
    from .utils.dummy_sentencepiece_objects import *

if is_tokenizers_available():
    from .tokenization_t5_fast import T5TokenizerFast
    from .tokenization_utils_fast import PreTrainedTokenizerFast

    if is_sentencepiece_available():
        from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, convert_slow_tokenizer
else:
    from .utils.dummy_tokenizers_objects import *

# Trainer
from .trainer_callback import (
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_utils import EvalPrediction, EvaluationStrategy, set_seed
from .training_args import TrainingArguments
from .training_args_tf import TFTrainingArguments
from .utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Modeling
if is_torch_available():
    from .generation_utils import top_k_top_p_filtering
    from .modeling_t5 import (
        T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        T5ForConditionalGeneration,
        T5Model,
        T5PreTrainedModel,
        load_tf_weights_in_t5,
    )
    from .modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer

    # Optimization
    from .optimization import (
        Adafactor,
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
    )

else:
    from .utils.dummy_pt_objects import *


if is_torch_available():
    logger.warning(
        "PyTorch >= 2.0 is not found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
