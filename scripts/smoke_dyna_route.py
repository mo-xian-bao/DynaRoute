import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dyna_route.models.modeling_dyna_route import DynaRouteConfig, DynaRouteForPrediction


def build_config(args):
    return DynaRouteConfig(
        input_size=1,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        horizon_lengths=[1, 2],
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_attention_heads,
        num_experts_per_tok=1,
        num_experts=args.num_experts,
        max_position_embeddings=args.max_position_embeddings,
        use_cache=True,
        use_dense=False,
        apply_aux_loss=True,
        use_dyna_route=True,
        dyna_route_codebook_size=args.codebook_size,
        dyna_route_code_dim=args.dynamics_dim,
        dyna_route_residual_dim=args.dynamics_dim,
        dyna_route_router_dim=args.dynamics_dim,
        _attn_implementation="eager",
    )


def main():
    parser = argparse.ArgumentParser("DynaRoute smoke test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--prediction_length", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--intermediate_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=8)
    parser.add_argument("--dynamics_dim", type=int, default=16)
    parser.add_argument("--max_position_embeddings", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(9899)
    model = DynaRouteForPrediction(build_config(args))
    input_ids = torch.randn(args.batch_size, args.seq_len)
    labels = torch.randn(args.batch_size, args.seq_len)
    loss_masks = torch.ones(args.batch_size, args.seq_len)

    model.train()
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        loss_masks=loss_masks,
        use_cache=False,
        return_dict=True,
    )
    outputs.loss.backward()
    print(f"train_loss={outputs.loss.detach().item():.6f}")
    print(f"logits_shape={tuple(outputs.logits.shape)}")
    print(f"dynamics_token_shape={tuple(outputs.dyna_route_token.shape)}")

    model.eval()
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=args.prediction_length)
    print(f"generated_shape={tuple(generated.shape)}")


if __name__ == "__main__":
    main()
