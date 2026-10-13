"""Spektrogram üzerinde maske (veya temiz genlik) tahmin eden 2B U-Net."""
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_MODES = ("mask", "direct")


class ConvBlock(nn.Module):
    """(Conv 3×3 → BatchNorm → LeakyReLU) × 2; uzamsal boyut değişmez."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """`depth` kez yarıya küçülten encoder, simetrik decoder ve isteğe bağlı skip bağlantıları.

    Seviye i'de kanal sayısı `base_ch * 2**i`, darboğazda `base_ch * 2**depth`.
    `skip=False` iken decoder'a skip yerine sıfır tensörü verilir: katmanlar ve parametre sayısı
    aynı kalır, yalnızca skip'in taşıdığı bilgi kaldırılır (E2 karşılaştırması için).
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 16, depth: int = 4, skip: bool = True, out_mode: str = "mask") -> None:
        super().__init__()
        if out_mode not in OUT_MODES:
            raise ValueError(f"out_mode {OUT_MODES} değerlerinden biri olmalı, verilen: {out_mode!r}")
        if depth < 1:
            raise ValueError("depth en az 1 olmalı")
        self.depth, self.skip, self.out_mode = depth, skip, out_mode
        chs = [base_ch * 2 ** i for i in range(depth + 1)]

        self.encoders = nn.ModuleList(ConvBlock(in_ch if i == 0 else chs[i - 1], chs[i]) for i in range(depth))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[depth - 1], chs[depth])
        self.ups = nn.ModuleList(nn.ConvTranspose2d(chs[i + 1], chs[i], kernel_size=2, stride=2) for i in reversed(range(depth)))
        self.decoders = nn.ModuleList(ConvBlock(2 * chs[i], chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, F, T). mask modunda [0,1] maske, direct modunda log1p genlik."""
        f, t = x.shape[-2:]
        multiple = 2 ** self.depth
        x = F.pad(x, (0, -t % multiple, 0, -f % multiple))  # F ve T'yi 2**depth katına sıfırla doldur

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip if self.skip else torch.zeros_like(skip)], dim=1))

        out = self.head(x)[..., :f, :t]  # dolguyu kırp
        return torch.sigmoid(out) if self.out_mode == "mask" else out
