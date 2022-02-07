import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm


from deepspeech_pytorch.decoder import Decoder, GreedyDecoder
from deepspeech_pytorch.validation import WordErrorRate
from deepspeech_pytorch.validation import CharErrorRate
from deepspeech_attack.attacks.fgsm import fgsm


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
            adv_inputs, targets, inputs_sizes, target_sizes = fgsm(adv_inputs, targets, inputs_sizes,
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
