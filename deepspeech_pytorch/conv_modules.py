import torch
import torch.nn as nn


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


def get_seq_lens(input_length, conv):
    """
    Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
    containing the size sequences that will be output by the network.
    :param input_length: 1D Tensor
    :return: 1D Tensor scaled by model
    """
    seq_len = input_length
    for m in conv.modules():
        if type(m) == nn.modules.conv.Conv2d:
            seq_len = torch.div(seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1,
                                m.stride[1],
                                rounding_mode='floor') + 1
            # seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
    return seq_len.int()


class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        ))

    def forward(self, x, lengths):
        x, _ = self.conv(x, lengths)
        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths


class CNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True)
        ))

    def forward(self, x, lengths):
        x, _ = self.conv(x, lengths)
        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths


class CNN4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(96, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True),
        ))

    def forward(self, x, lengths):
        x, _ = self.conv(x, lengths)
        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths


class SkipOneStep2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )
        self.conv2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.projection1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.projection2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.conv = nn.Sequential(
            self.conv1, self.conv2
        )

    def forward(self, x, lengths):
        out1, _ = self.conv1(x, lengths)
        projected_x, _ = self.projection1(x, lengths)
        x = projected_x + out1

        out2, _ = self.conv2(x, lengths)
        projected_x, _ = self.projection2(x, lengths)
        x = projected_x + out2
        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths


class SkipOneStep2Runge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )
        self.conv2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.projection1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.projection2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.projection3 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(4, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.conv = nn.Sequential(
            self.conv1, self.conv2, self.projection3
        )

    def forward(self, x, lengths):
        out1, _ = self.conv1(x, lengths)
        projected_x, _ = self.projection1(x, lengths)
        x = projected_x + out1

        out2, _ = self.conv2(x, lengths)
        projected_x, _ = self.projection2(x, lengths)
        out3 = projected_x + out2 / 2

        out4, _ = self.conv2(out3, lengths)
        projected_x, _ = self.projection3(x, lengths)
        x = projected_x + out4

        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths


class SkipOneStep3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )
        self.conv2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )
        self.conv3 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.projection1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.projection2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.projection3 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.conv = nn.Sequential(
            self.conv1, self.conv2, self.conv3
        )

    def forward(self, x, lengths):
        out1, _ = self.conv1(x, lengths)
        projected_x, _ = self.projection1(x, lengths)
        x = projected_x + out1

        out2, _ = self.conv2(x, lengths)
        projected_x, _ = self.projection2(x, lengths)
        x = projected_x + out2

        out3, _ = self.conv3(x, lengths)
        projected_x, _ = self.projection3(x, lengths)
        x = projected_x + out3

        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths


class SkipTwoStep2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.projection = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 1), stride=(4, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

    def forward(self, x, lengths):
        out, _ = self.conv(x, lengths)
        projected_x, _ = self.projection(x, lengths)
        x = out + projected_x

        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths


class SkipOneStep4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )
        self.conv2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )
        self.conv3 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True))
        )
        self.conv4 = MaskConv(nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.projection1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.projection2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.projection3 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.projection4 = MaskConv(nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.conv = nn.Sequential(
            self.conv1, self.conv2, self.conv3, self.conv4
        )

    def forward(self, x, lengths):
        out1, _ = self.conv1(x, lengths)
        projected_x, _ = self.projection1(x, lengths)
        x = projected_x + out1

        out2, _ = self.conv2(x, lengths)
        projected_x, _ = self.projection2(x, lengths)
        x = projected_x + out2

        out3, _ = self.conv3(x, lengths)
        projected_x, _ = self.projection3(x, lengths)
        x = projected_x + out3

        out4, _ = self.conv4(x, lengths)
        projected_x, _ = self.projection4(x, lengths)
        x = projected_x + out4

        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths


class SkipTwoStep4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.conv2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(96, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.projection1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 1), stride=(4, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.projection2 = MaskConv(nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=(1, 1), stride=(4, 1)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.conv = nn.Sequential(
            self.conv1, self.conv2
        )

    def forward(self, x, lengths):
        out1, _ = self.conv1(x, lengths)
        projected_x, _ = self.projection1(x, lengths)
        x = projected_x + out1

        out2, _ = self.conv2(x, lengths)
        projected_x, _ = self.projection2(x, lengths)
        x = projected_x + out2

        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths


class DenseStep1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )
        self.conv2 = MaskConv(nn.Sequential(
            nn.Conv2d(32 + 1, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True))
        )

        self.projection1 = MaskConv(nn.Sequential(
            nn.AvgPool2d(kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.projection2 = MaskConv(nn.Sequential(
            nn.AvgPool2d(kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32 + 1),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        self.conv = nn.Sequential(self.conv1, self.conv2)

    def forward(self, x, lengths):
        out1, _ = self.conv1(x, lengths)
        projected_x, _ = self.projection1(x, lengths)
        x = torch.cat([projected_x, out1], dim=1)  # dim 1 + 32 = 33

        out2, _ = self.conv2(x, lengths)
        projected_x, _ = self.projection2(x, lengths)
        x = torch.cat([projected_x, out2], dim=1)  # dim 33 + 32 = 65

        output_lengths = get_seq_lens(lengths, self.conv)
        return x, output_lengths
