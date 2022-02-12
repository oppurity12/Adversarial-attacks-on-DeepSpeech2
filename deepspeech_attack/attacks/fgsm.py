import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm


from deepspeech_pytorch.decoder import Decoder, GreedyDecoder
from deepspeech_pytorch.validation import WordErrorRate
from deepspeech_pytorch.validation import CharErrorRate

from copy import deepcopy


def is_correct(
        preds: torch.Tensor,
        preds_sizes: torch.Tensor,
        targets: torch.Tensor,
        target_sizes: torch.Tensor,
        wer: WordErrorRate,
        cer: CharErrorRate):
    # unflatten targets
    split_targets = []
    offset = 0
    decoder = wer.decoder
    target_decoder = wer.target_decoder

    for size in target_sizes:
        split_targets.append(targets[offset:offset + size])
        offset += size

    decoded_output, _ = decoder.decode(preds, preds_sizes)
    target_strings = target_decoder.convert_to_strings(split_targets)
    for x in range(len(target_strings)):
        transcript, reference = decoded_output[x][0], target_strings[x][0]
        if wer.wer_calc(transcript, reference) == 0 and cer.cer_calc(transcript, reference) == 0:
            return True


def fgsm(inputs, targets, inputs_sizes, target_sizes, model, criterion, device, epsilon):
    model.train()
    inputs = inputs.detach()
    inputs = inputs.to(device)
    inputs.requires_grad = True
    out, output_sizes = model(inputs, inputs_sizes)
    # decoded_output, _ = decoder.decode(out, output_sizes)
    out = out.transpose(0, 1)  # TxNxH
    out = out.log_softmax(-1)
    model.zero_grad()
    loss = criterion(out, targets, output_sizes, target_sizes)
    loss.backward()

    perturbation = epsilon * inputs.grad.sign()
    adv_inputs = inputs + perturbation
    adv_inputs = torch.clamp(adv_inputs, min=-1, max=10)
    return adv_inputs, targets, inputs_sizes, target_sizes


def fg_main(test_loader,
           model,
           test_model,
           decoder: Decoder,
           device: torch.device,
           target_decoder: Decoder,
           precision: int,
           cfg):
    wer = WordErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    cer = CharErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    try:
        epsilon = cfg.epsilon
    except AttributeError:
        raise AttributeError("Need epsilon flag")

    labels = ["_", "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
              "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
              "Y", "Z", " "]

    criterion = nn.CTCLoss(blank=labels.index('_'), reduction='sum', zero_infinity=True)

    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = batch
        inputs_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        adv_inputs, targets, inputs_sizes, target_sizes = fgsm(inputs, targets, inputs_sizes,
                                                               target_sizes, model, criterion,
                                                               device, epsilon)

        with autocast(enabled=precision == 16) and torch.no_grad():
            test_model.eval()
            adv_out, adv_output_sizes = test_model(adv_inputs.to(device), inputs_sizes)

        # decoded_output, _ = decoder.decode(adv_out, adv_output_sizes)
        wer.update(
            preds=adv_out,
            preds_sizes=adv_output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        cer.update(
            preds=adv_out,
            preds_sizes=adv_output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )

    return wer.compute(), cer.compute()
