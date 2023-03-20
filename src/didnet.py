from keras.layers import Layer, GlobalAveragePooling2D, Conv2D, ReLU, Activation, Multiply, Add, BatchNormalization
from keras.models import Sequential
from keras.utils import plot_model

# CAL and RCAB are defined in: https://github.com/yulunzhang/RCAN
class ChannelAttentionLayer(Layer):
    def __init__(self, output_dim, channel, reduction, **kwargs):
        super(ChannelAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.channel = channel
        self.reduction = reduction

    # TODO: Verify the valid of the padding value
    def build(self, input_shape):
        self.body = Sequential(layers=[
            GlobalAveragePooling2D(keepdims=True),
            Conv2D(filters=self.channel//self.reduction,
                   kernel_size=1, padding="same", use_bias=True),
            ReLU(),
            Conv2D(filters=self.channel, kernel_size=1,
                   padding="same", use_bias=True),
            Activation("sigmoid")
        ], name="CAL")
        super(ChannelAttentionLayer, self).build(input_shape)

    def call(self, input_data):
        res = self.body(input_data)
        return Multiply()([res, input_data])

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
                Conv2D(filters=self.n_feat, kernel_size=self.kernel_size, padding="same", use_bias=True))
            if self.bn:
                modules_body.append(BatchNormalization())
            if i == 0:
                modules_body.append(ReLU())
        modules_body.append(ChannelAttentionLayer(
            output_dim=self.output_dim, channel=self.n_feat, reduction=self.reduction))
        # TODO: Verify this one, it isn't in the original implementation
        modules_body.append(
            Conv2D(filters=3, kernel_size=self.kernel_size, padding="same", use_bias=True))
        self.body = Sequential(layers=modules_body, name="RCAB")
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

        self.body = Sequential(layers=modules, name="BM")
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
        # Add 1x1 conv layer
        self.body = Sequential(layers=[
            BodyModule(output_dim=self.output_dim),
            # TODO: Verifi if the number of filters is right
            Conv2D(filters=64, kernel_size=1, activation='relu')
        ], name="UB")
        super(UpsamplingBlock, self).build(input_shape)

    def call(self, input_data):
        return self.body(input_data)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def main():
    g = Sequential(name="G_Model")
    g.add(UpsamplingBlock(32, input_shape=(128, 128, 3)))
    print(g.summary(expand_nested=True))


if __name__ == "__main__":
    main()
