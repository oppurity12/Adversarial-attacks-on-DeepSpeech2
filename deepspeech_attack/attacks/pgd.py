import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm


from deepspeech_pytorch.decoder import Decoder, GreedyDecoder
from deepspeech_pytorch.validation import WordErrorRate
from deepspeech_pytorch.validation import CharErrorRate


def pgd(inputs, targets, inputs_sizes, target_sizes, model, criterion, device, epsilon=0.1, alpha=1):
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

    perturbation = alpha * inputs.grad.sign()
    adv_inputs = inputs + perturbation
    delta = torch.clamp(adv_inputs - inputs, min=-epsilon, max=epsilon)
    adv_inputs = torch.clamp(inputs + delta, min=-1, max=10)
    return adv_inputs, targets, inputs_sizes, target_sizes


def pgd_main(test_loader,
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
        attack_steps = cfg.attack_steps
    except AttributeError:
        raise AttributeError("Need epsilon flag and cfg.attack_steps flag")

    labels = ["_", "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
              "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
              "Y", "Z", " "]

    criterion = nn.CTCLoss(blank=labels.index('_'), reduction='sum', zero_infinity=True)
    model.train()
    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = batch
        inputs_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        adv_inputs = inputs
        for step in range(attack_steps):
            adv_inputs, targets, inputs_sizes, target_sizes = pgd(adv_inputs, targets, inputs_sizes,
                                                                   target_sizes, model, criterion,
                                                                   device, epsilon)

        with autocast(enabled=precision == 16) and torch.no_grad():
            test_model.eval()
            adv_out, adv_output_sizes = test_model(adv_inputs, inputs_sizes)
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
