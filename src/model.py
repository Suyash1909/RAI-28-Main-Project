import torch
import torch.nn as nn

BOXES_PER_CELL = 3
NUM_CLASSES = 3
VALUES_PER_BOX = 5 + NUM_CLASSES


class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = dilation  # Preserves exact spatial height and width

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class potholecnn(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor with dilated convolutions for full-image context
        self.features = nn.Sequential(
            convblock(3, 16, dilation=1),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            convblock(16, 32, dilation=1),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            convblock(32, 64, dilation=2),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            convblock(64, 128, dilation=2),
            convblock(128, 128, dilation=4),  # Wide receptive field for giant potholes
        )

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

        # Learn spatial features after upsampling (32x32)
        self.refine = nn.Sequential(
            convblock(128, 128, dilation=1),
            convblock(128, 128, dilation=2),
        )

        self.detection_head = nn.Conv2d(
            in_channels=128,
            out_channels=BOXES_PER_CELL * VALUES_PER_BOX,
            kernel_size=1,
        )

        # Initialize objectness logit bias to -4.0 (initial confidence ~0.018)
        with torch.no_grad():
            bias = self.detection_head.bias.view(
                BOXES_PER_CELL, VALUES_PER_BOX
            )
            bias[:, 4].fill_(-4.0)

    def forward(self, x):
        features = self.features(x)
        features = self.upsample(features)
        features = self.refine(features)

        predictions = self.detection_head(features)

        batch_size = predictions.shape[0]

        predictions = predictions.reshape(
            batch_size, BOXES_PER_CELL, VALUES_PER_BOX, 32, 32
        )

        predictions = predictions.permute(0, 3, 4, 1, 2)

        return predictions


if __name__ == "__main__":
    model = potholecnn()
    model.eval()

    x = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        predictions = model(x)

    print("Input shape:", x.shape)
    print("Prediction shape:", predictions.shape)
    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters()),
    )