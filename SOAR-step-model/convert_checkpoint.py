import argparse
import os

import torch
import pickle


class _StubClass:
    """Placeholder for classes whose defining module isn't available (e.g. the
    original LoViT ``models`` package), so pickled objects can still be
    unpickled far enough to recover their tensors."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(f"{module}.{name}", (_StubClass,), {})

class _tolerant_pickle_module:
    Unpickler = _TolerantUnpickler
    Pickler = pickle.Pickler
    load = pickle.load


def _flatten_module(obj, prefix=""):
    """Recursively walk an (possibly stubbed) nn.Module-like object's
    _parameters/_buffers/_modules and collect a flat {name: tensor} dict,
    mirroring what nn.Module.state_dict() would produce."""
    flat = {}
    for name, tensor in (getattr(obj, "_parameters", None) or {}).items():
        if tensor is not None:
            flat[prefix + name] = tensor.detach()
    for name, tensor in (getattr(obj, "_buffers", None) or {}).items():
        if tensor is not None:
            flat[prefix + name] = tensor
    for name, submodule in (getattr(obj, "_modules", None) or {}).items():
        if submodule is not None:
            flat.update(_flatten_module(submodule, prefix + name + "."))
    return flat


def load_state_dict(path):
    """Load a checkpoint and return a flat {name: tensor} dict, whether it was
    saved as a plain state_dict, a wrapper dict containing one, or a full
    pickled model object."""
    ckpt = torch.load(path, map_location="cpu", pickle_module=_tolerant_pickle_module, weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt)
        return {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    return _flatten_module(ckpt)


def convert(src_path, dst_path):
    state_dict = load_state_dict(src_path)
    torch.save({"state_dict": state_dict}, dst_path)
    print(f"Wrote {len(state_dict)} tensors to {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a full pickled LoViT model checkpoint (e.g. Trained_VIT_Cholec80.pth) "
        "into the {'state_dict', 'use_se'} format expected as step_encoder.pth."
    )
    parser.add_argument("--src", default="Trained_VIT_Cholec80.pth", help="Source checkpoint to convert.")
    parser.add_argument("--dst", default="models/step_encoder.pth", help="Output path.")
    parser.add_argument("--force", action="store_true", help="Overwrite --dst if it already exists.")
    args = parser.parse_args()

    if os.path.exists(args.dst) and not args.force:
        raise SystemExit(f"{args.dst} already exists; pass --force to overwrite.")

    convert(args.src, args.dst)
