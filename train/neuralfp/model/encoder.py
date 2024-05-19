import torch.nn as nn


class SepConvEncoder(nn.Module):
    def __init__(self, d, h, in_F, segment_size, stft_hop, sample_rate):
        super(SepConvEncoder, self).__init__()
        in_T = (int(segment_size * sample_rate) + stft_hop - 1) // stft_hop
        channels = [1, d, d, 2 * d, 2 * d, 4 * d, 4 * d, h, h]
        convs = []
        for i in range(8):
            k = 3
            s = 2, 2
            block = SepConvBlock(channels[i], channels[i + 1], k, s, in_F, in_T)
            convs.append(block)
            in_F = (in_F - 1) // s[1] + 1
            in_T = (in_T - 1) // s[0] + 1
        assert in_F == in_T == 1, "output must be 1x1"
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
        return x


class SepConvBlock(nn.Module):
    def __init__(self, i, o, k, s, in_F, in_T):
        super(SepConvBlock, self).__init__()
        padding = (in_T - 1) // s[0] * s[0] + k - in_T
        self.pad1 = nn.ZeroPad2d((padding // 2, padding - padding // 2, 0, 0))
        self.conv1 = nn.Conv2d(i, o, kernel_size=(1, k), stride=(1, s[0]))
        self.ln1 = nn.LayerNorm((o, in_F, (in_T - 1) // s[0] + 1))
        self.relu1 = nn.ReLU()

        padding = (in_F - 1) // s[1] * s[1] + k - in_F
        self.pad2 = nn.ZeroPad2d((0, 0, padding // 2, padding - padding // 2))
        self.conv2 = nn.Conv2d(o, o, kernel_size=(k, 1), stride=(s[1], 1), groups=o)
        self.ln2 = nn.LayerNorm((o, (in_F - 1) // s[1] + 1, (in_T - 1) // s[0] + 1))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        return x
