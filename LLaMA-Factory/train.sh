
FORCE_TORCHRUN=1 llamafactory-cli train exp/MATH_short_CoT.yaml

FORCE_TORCHRUN=1 llamafactory-cli train exp/MATH_long_CoT.yaml

FORCE_TORCHRUN=1 llamafactory-cli train exp/MATH_mix.yaml

