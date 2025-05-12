import torch

def vit_forward(self, x):
# vanilla vit
# input: x (B, S, D)
# transformer encoder with L blocks
    for i, block in enumerate(self.blocks):
        # forward input through each block
        x = block(x)

    x = self.head(x)

    return x


def vit_forward_w_ila(self, x):
# vit with ILA modules
# input: x (B, S, D)
# transformer encoder with L blocks
    j = 0
    for i, block in enumerate(self.blocks):
        # forward input through each block
        x = block(x)

        # ila_locs: list with locs for
        # downsampling with ILA module
        # eg: self.ila_locs = [3, 7]
        if i in self.ila_locs:
            x = self.ila[j](x)
            j += 1

    x = self.head(x)

    return x


def forward_ila(self, x):
    # x: shape B, S, D
    res = x

    # channel downsampling (cds)
    x = self.cds_linear(x)
    x = self.cds_bn(x)
    x = self.act(x)
    # x: B, S, K (K << D)

    # spatial downsampling (sds)
    x = rearrange(x, 'b (fh1 fw1) k -> b k fh1 fw1')
    x = self.sds_dwconv(x)
    # no padding so fh1 and fw1 -> fh2 and fw2
    # x: b k fh2 fw2
    x = self.sds_bn(x)
    x = self.act(x)

    x = self.sds_pwconv(x)
    x = self.sds_bn(x)
    x = self.act(x)
    x = rearrange(x, 'b k fh2 fw2 -> b (fh2 fw2) k')

    # channel upsampling (cus)
    x = self.cus_linear(x)
    # x: b, s_ds, d

    # residual spatial downsampling (rsds)
    # 1st: why is residual / identity f important? 
    # refer to resnet and other papers
    # 2nd: naive solutions to downsample:
    # avg/max pool, linear/bilinear interpolation
    # 3rd: we propose an alternative:
    # learnable downsampling (depthwise convolutions)
    # reason: same reason why we moved from
    # fixed kernels/filters in traditional CV 
    # to learnable kernels (CNNs, AlexNet)
    # allows for the network to learn an optimal
    # set of weights (to downsample)
    # 4th: why depthwise convolution
    # (instead of traditional convolution)
    # traditional convolution is difficult to learn
    # an identity function: kernel_size = 1
    # in the case of kernel size of 1
    # based on this intuition 
    # (possibility of the dwconv to model an identity function)
    # we extend this to other kernel sizes
    # ablation on this
    res = self.rsds_dwconv(res)
    x = x + res

    return x