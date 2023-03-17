from keras.layers import Layer, GlobalAveragePooling2D, Conv2D, ReLU, Activation, Multiply, Add, BatchNormalization
from keras.models import Sequential


# CAL and RCAB are defined in: https://github.com/yulunzhang/RCAN
class ChannelAttentionLayer(Layer):
    def __init__(self, output_dim, channel, reduction, **kwargs):
        super(ChannelAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.channel = channel
        self.reduction = reduction

    # TODO: Verifiy if input_size should be passed as arg. Also verify the corresponding padding value
    def build(self, input_shape):
        self.avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.conv_du = Sequential(layers=[
            Conv2D(filters=self.channel//self.reduction,
                   kernel_size=1, padding="valid", use_bias=True),
            ReLU(),
            Conv2D(filters=self.channel, kernel_size=1,
                   padding="valid", use_bias=True),
            Activation("sigmoid")
        ])
        super(ChannelAttentionLayer, self).build(input_shape)

    def call(self, input_data):
        y = self.avg_pool(input_data)
        y = self.conv_du(y)
        return Multiply()([input_data, y])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class ResidualChannelAttentionBlock(Layer):
    def __init__(self, output_dim,
                 n_feat=64, kernel_size=3, reduction=16,
                 bias=True, bn=False, res_scale=1, **kwargs):
        super(ResidualChannelAttentionBlock, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.n_feat = n_feat
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.bias = bias
        self.bn = bn
        self.res_scale = res_scale

    def build(self, input_shape):
        modules_body = []
        for i in range(2):
            modules_body.append(
                Conv2D(filters=self.n_feat, kernel_size=self.kernel_size, padding="valid", use_bias=True))
            if self.bn:
                modules_body.append(BatchNormalization())
            if i == 0:
                modules_body.append(ReLU())
        modules_body.append(ChannelAttentionLayer(
            output_dim=self.output_dim, channel=self.n_feat, reduction=self.reduction))
        self.body = Sequential(layers=modules_body)
        super(ResidualChannelAttentionBlock, self).build(input_shape)

    def call(self, input_data):
        res = self.body(input_data)
        return Add()([res, input_data])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class BodyModule(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(BodyModule, self).__init__(**kwargs)

    def build(self, input_shape):
        modules = []
        for _ in range(30):
            modules.append(ResidualChannelAttentionBlock(
                output_dim=self.output_dim))

        self.body = Sequential(layers=modules)
        super(BodyModule, self).build(input_shape)

    def call(self, input_data):
        return self.body(input_data)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class UpsamplingBlock(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(UpsamplingBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add 1 body module
        # Add 1 upsampling module
        # Add 1x1 con layer
        self.body = Sequential(layers=[
            BodyModule(output_dim=self.output_dim)
        ])
        super(UpsamplingBlock, self).build(input_shape)

    def call(self, input_data):
        return self.body(input_data)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def main():
    g = Sequential()
    g.add(UpsamplingBlock(32, input_shape=(64, 64, 3)))


if __name__ == "__main__":
    main()
