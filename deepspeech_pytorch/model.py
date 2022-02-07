import math
from typing import List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn import CTCLoss

from deepspeech_pytorch.configs.train_config import SpectConfig, BiDirectionalConfig, OptimConfig, AdamConfig, \
    SGDConfig, UniDirectionalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.validation import CharErrorRate, WordErrorRate
from deepspeech_pytorch.conv_modules import CNN2, CNN3, CNN4, SkipOneStep2, SkipOneStep3, \
    SkipOneStep4, SkipTwoStep2, SkipTwoStep4, SkipOneStep2Runge, DenseStep1

from deepspeech_pytorch.drnn import DRNN
from deepspeech_pytorch.residual_lstm import ResLSTMLayer


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


# uni_direction
class BatchDilatedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_norm=True):
        super(BatchDilatedRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = DRNN(n_input=input_size, n_hidden=hidden_size, n_layers=1)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        # x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        # x, _ = nn.utils.rnn.pad_packed_sequence(x)
        return x


class BatchResidualLstm(nn.Module):
    def __init__(self, input_size, hidden_size, batch_norm=True):
        super(BatchResidualLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = ResLSTMLayer(input_size=input_size, hidden_size=hidden_size)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        batch_size = x.shape[1]
        # x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        hx, cx = torch.zeros((1, batch_size, self.hidden_size)).cuda(), torch.zeros((1, batch_size, self.hidden_size)).cuda()
        hidden = (hx, cx)
        x, h = self.rnn(x, hidden)
        # x, _ = nn.utils.rnn.pad_packed_sequence(x)
        return x


class BatchResRNN(BatchRNN):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super().__init__(input_size=input_size, hidden_size=hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=batch_norm)

    def forward(self, x, output_lengths):
        out1 = super().forward(x, output_lengths)
        out2 = super().forward(out1, output_lengths)
        res = x + out2
        return res


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.context,
            stride=1,
            groups=self.n_features,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(pl.LightningModule):
    def __init__(self,
                 labels: List,
                 model_cfg: Union[UniDirectionalConfig, BiDirectionalConfig],
                 precision: int,
                 optim_cfg: Union[AdamConfig, SGDConfig],
                 spect_cfg: SpectConfig,
                 number_of_layers: int,
                 skip_steps: int
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.precision = precision
        self.optim_cfg = optim_cfg
        self.spect_cfg = spect_cfg
        self.bidirectional = True if OmegaConf.get_type(model_cfg) is BiDirectionalConfig else False
        self.number_of_layers = number_of_layers
        self.skip_steps = skip_steps

        self.labels = labels
        num_classes = len(self.labels)

        conv_lists = {2: [CNN2(), SkipOneStep2(), SkipTwoStep2()], 3: [CNN3(), SkipOneStep3()],
                      4: [CNN4(), SkipOneStep4(), SkipTwoStep4()]}
        self.conv = conv_lists[number_of_layers][self.skip_steps]
        # self.conv = SkipOneStep2Runge()
        # self.conv = DenseStep1()

        # self.conv = MaskConv(nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
        #     nn.BatchNorm2d(32),
        #     nn.Hardtanh(0, 20, inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
        #     nn.BatchNorm2d(32),
        #     nn.Hardtanh(0, 20, inplace=True)
        # ))

        # if self.shortcut:
        #     self.projection = MaskConv(nn.Sequential(
        #         nn.Conv2d(1, 32, kernel_size=(1, 1), stride=(4, 2)),
        #         nn.BatchNorm2d(32),
        #         nn.Hardtanh(0, 20, inplace=True)
        #     ))

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        # rnn_input_size = int(math.floor((self.spect_cfg.sample_rate * self.spect_cfg.window_size) / 2) + 1)
        # rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        # rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        # rnn_input_size *= 32

        rnn_input_size = self.get_rnn_input_size(self.conv.conv)

        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=self.model_cfg.hidden_size,
                rnn_type=self.model_cfg.rnn_type.value,
                bidirectional=self.bidirectional,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=self.model_cfg.hidden_size,
                    hidden_size=self.model_cfg.hidden_size,
                    rnn_type=self.model_cfg.rnn_type.value,
                    bidirectional=self.bidirectional
                ) for x in range(self.model_cfg.hidden_layers - 1)
            )
        )

        # self.rnns = nn.Sequential(
        #     BatchResidualLstm(
        #         input_size=rnn_input_size,
        #         hidden_size=self.model_cfg.hidden_size,
        #         batch_norm=False
        #     ),
        #     *(
        #         BatchResidualLstm(
        #             input_size=self.model_cfg.hidden_size,
        #             hidden_size=self.model_cfg.hidden_size,
        #         ) for x in range(self.model_cfg.hidden_layers - 1)
        #     )
        # )

        # self.rnns = nn.Sequential(
        #     BatchDilatedRNN(
        #         input_size=rnn_input_size,
        #         hidden_size=self.model_cfg.hidden_size,
        #         batch_norm=False,
        #     ),
        #     *(
        #         BatchDilatedRNN(
        #             input_size=self.model_cfg.hidden_size,
        #             hidden_size=self.model_cfg.hidden_size,
        #         ) for x in range(self.model_cfg.hidden_layers - 1)
        #     )
        # )

        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(self.model_cfg.hidden_size, context=self.model_cfg.lookahead_context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.model_cfg.hidden_size),
            nn.Linear(self.model_cfg.hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()
        self.criterion = CTCLoss(blank=self.labels.index('_'), reduction='sum', zero_infinity=True)
        self.evaluation_decoder = GreedyDecoder(self.labels)  # Decoder used for validation
        self.wer = WordErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )
        self.cer = CharErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        x, output_lengths = self.conv(x, lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def training_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        out, output_sizes = self(inputs, input_sizes)
        out = out.transpose(0, 1)  # TxNxH
        out = out.log_softmax(-1)

        loss = self.criterion(out, targets, output_sizes, target_sizes)
        self.log('loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(self.device)
        with autocast(enabled=self.precision == 16):
            out, output_sizes = self(inputs, input_sizes)
        decoded_output, _ = self.evaluation_decoder.decode(out, output_sizes)
        self.wer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        self.cer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        self.log('wer', self.wer.compute(), prog_bar=True, on_epoch=True)
        self.log('cer', self.cer.compute(), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        if OmegaConf.get_type(self.optim_cfg) is SGDConfig:
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                momentum=self.optim_cfg.momentum,
                nesterov=True,
                weight_decay=self.optim_cfg.weight_decay
            )
        elif OmegaConf.get_type(self.optim_cfg) is AdamConfig:
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                betas=self.optim_cfg.betas,
                eps=self.optim_cfg.eps,
                weight_decay=self.optim_cfg.weight_decay
            )
        else:
            raise ValueError("Optimizer has not been specified correctly.")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.optim_cfg.learning_anneal
        )
        return [optimizer], [scheduler]

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = torch.div(seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1,
                                    m.stride[1],
                                    rounding_mode='floor') + 1
        return seq_len.int()

    def get_rnn_input_size(self, conv):
        rnn_input_size = int(math.floor((self.spect_cfg.sample_rate * self.spect_cfg.window_size) / 2) + 1)
        last_output_channel = 1
        for m in conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                rnn_input_size = int(math.floor(rnn_input_size + 2 * m.padding[0] - m.kernel_size[0]) / 2 + 1)
                last_output_channel = m.out_channels

        return rnn_input_size * last_output_channel


