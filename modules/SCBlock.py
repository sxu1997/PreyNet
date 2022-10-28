import torch
import torch.nn as nn


# x 是特征， y是单通道引导
class SCBlock(nn.Module):
    def __init__(self, channel, subchannel):
        super(SCBlock, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            raise Exception("Invalid Channel")

        result = self.conv(x_cat)
        # x = x + self.conv(x_cat)
        # y = y + self.score(x)

        return result

# Group-Reversal Attention (GRA) Block
# class GRA(nn.Module):
#     def __init__(self, channel, subchannel):
#         super(GRA, self).__init__()
#         self.group = channel//subchannel
#         self.conv = nn.Sequential(
#             nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
#         )
#         self.score = nn.Conv2d(channel, 1, 3, padding=1)
#
#     def forward(self, x, y):
#         if self.group == 1:
#             x_cat = torch.cat((x, y), 1)
#         elif self.group == 2:
#             xs = torch.chunk(x, 2, dim=1)
#             x_cat = torch.cat((xs[0], y, xs[1], y), 1)
#         elif self.group == 4:
#             xs = torch.chunk(x, 4, dim=1)
#             x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
#         elif self.group == 8:
#             xs = torch.chunk(x, 8, dim=1)
#             x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
#         elif self.group == 16:
#             xs = torch.chunk(x, 16, dim=1)
#             x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
#             xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
#         elif self.group == 32:
#             xs = torch.chunk(x, 32, dim=1)
#             x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
#             xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
#             xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
#             xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
#         else:
#             raise Exception("Invalid Channel")
#
#         x = x + self.conv(x_cat)
#         y = y + self.score(x)
#
#         return x, y