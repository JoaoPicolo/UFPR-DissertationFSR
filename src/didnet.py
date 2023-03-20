from keras.layers import Layer, GlobalAveragePooling2D, Conv2D, ReLU, Activation, Multiply, Add, BatchNormalization
from keras import Input, Model, Sequential

# CAL and RCAB are defined in: https://github.com/yulunzhang/RCAN
class ChannelAttentionLayer(Layer):
    def __init__(self, channel, reduction, **kwargs):
        super(ChannelAttentionLayer, self).__init__(**kwargs)
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


class ResidualChannelAttentionBlock(Layer):
    def __init__(self, n_feat=64, kernel_size=3, reduction=16,
                 bias=True, bn=False, res_scale=1, **kwargs):
        super(ResidualChannelAttentionBlock, self).__init__(**kwargs)
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
        modules_body.append(ChannelAttentionLayer(channel=self.n_feat, reduction=self.reduction))
        # TODO: Verify this one, it isn't in the original implementation
        modules_body.append(
            Conv2D(filters=3, kernel_size=self.kernel_size, padding="same", use_bias=True))
        self.body = Sequential(layers=modules_body, name="RCAB")
        super(ResidualChannelAttentionBlock, self).build(input_shape)

    def call(self, input_data):
        res = self.body(input_data)
        return Add()([res, input_data])


class BodyModule(Layer):
    def __init__(self, **kwargs):
        super(BodyModule, self).__init__(**kwargs)

    def build(self, input_shape):
        modules = []
        for _ in range(30):
            modules.append(ResidualChannelAttentionBlock())

        self.body = Sequential(layers=modules, name="BM")
        super(BodyModule, self).build(input_shape)

    def call(self, input_data):
        return self.body(input_data)
    

class UpsamplingModule(Layer):
    def __init__(self, **kwargs):
        super(UpsamplingModule, self).__init__(**kwargs)

    # Pixel-shuffle references this: https://keras.io/examples/vision/super_resolution_sub_pixel/#build-a-model
    def build(self, input_shape):
        self.body = Sequential(layers=[
            # Pixel-shuffle
            Conv2D(filters=64, kernel_size=5, activation='relu', padding="same", kernel_initializer="Orthogonal"),
            Conv2D(filters=64, kernel_size=3, activation='relu', padding="same", kernel_initializer="Orthogonal"),
            Conv2D(filters=32, kernel_size=3, activation='relu', padding="same", kernel_initializer="Orthogonal"),
            Conv2D(filters=9,  kernel_size=3, activation='relu', padding="same", kernel_initializer="Orthogonal"),
            # Conv 3x3
            Conv2D(filters=3, kernel_size=3, activation='relu')
        ], name="UM")
        super(UpsamplingModule, self).build(input_shape)

    def call(self, input_data):
        return self.body(input_data)


class UpsamplingBlock(Layer):
    def __init__(self, **kwargs):
        super(UpsamplingBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.body = Sequential(layers=[
            BodyModule(),
            UpsamplingModule(),
            # TODO: Verify if the number of filters is right
            Conv2D(filters=3, kernel_size=1, activation='relu')
        ], name="UB")
        super(UpsamplingBlock, self).build(input_shape)

    def call(self, input_data):
        return self.body(input_data)
    

def getGModel():
    inputs = Input(shape=(128,128,3))
    outputs = UpsamplingBlock()(inputs)
    # outputs = UpsamplingBlock()(outputs)
    model = Model(inputs, outputs)
    return model

def main():
    g = getGModel()
    print(g.summary())


if __name__ == "__main__":
    main()
