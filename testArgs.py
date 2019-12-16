import sys
import argparse


# NN Training:
 "optimizer": RAdam,
# NN architecture setup:
"layers_feature_vector": layers_feature_vector, "layers_state_action_merge": layers_state_action_merge,
"layers_action": layers_action,
"layers_feature_merge": layers_feature_merge, "layers_r": layers_r, "layers_Q": layers_Q,
"layers_V": layers_V,
"layers_actor": layers_actor,
"layers_feature_matrix": conv_mnhi_later,

# Env specific:
"convert_2_torch_wrapper": None,
"action_wrapper": None,
"always_keys": ["sprint"], "exclude_keys": ["sneak"], "reverse_keys": ["forward"],
"use_MineRL_policy": False,
"forward_when_jump": True,

args = parser.parse_args()

arg_dict = vars(args)

for arg in arg_dict:
    print(arg, arg_dict[arg])