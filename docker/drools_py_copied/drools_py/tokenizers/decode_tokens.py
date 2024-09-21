import torch


def do_get_to_decode_hf(logits, output_ids=None):
    if output_ids is None:
        if len(logits.shape) == 3:
            detached_model_outputs = logits.data.clone().detach()
            to_decode = torch.argmax(detached_model_outputs, dim=2)
        else:
            to_decode = logits
    else:
        to_decode = output_ids
    if len(to_decode.shape) > 1:
        to_decode = to_decode.transpose(0, 1)
    return to_decode
