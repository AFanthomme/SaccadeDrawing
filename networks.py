import torch as tch
import logging

class RNNCell(tch.nn.Module):
    def __init__(self, n_in, n_rec):
        super().__init__(n_in, n_rec)
        self.rnn_rec = tch.nn.Linear(n_rec, n_rec)
        self.rnn_in = tch.nn.Linear(n_in, n_rec)
        self.rnn_out = tch.nn.Linear(n_rec, n_rec)

    def forward(self, x, h):
        return tch.relu(self.rnn_rec(h) + self.rnn_in(x))


class ConvolutionalRecurrentNetwork(tch.nn.Module):
    # Recurrence can a priori help with visual processing, so define a model that has 3 parts:
    def __init__(self, architecture_specs):
        super().__init__()
        self.__dict__.update(architecture_specs)
        
        # Build the convnet
        conv_layers = []
        scale_factor = 1
        n_cnn_layers = len(self.cnn_n_featuremaps) - 1
        i = 0
        for c_in, c_out, kernel_width, kernel_stride in zip([3] + self.cnn_n_featuremaps[:-1], self.cnn_n_featuremaps, self.cnn_kernel_sizes, self.cnn_kernel_strides):
            conv_layers.append(tch.nn.Conv2d(c_in, c_out, kernel_width, stride=kernel_stride, padding='same'))
            conv_layers.append(tch.nn.ReLU())
            if i < n_cnn_layers - 1:
                conv_layers.append(tch.nn.MaxPool2d(2))
                scale_factor *= 2
            i += 1

        # Want 4x4 feature maps after CNN so 128 of them = 2048 vector.
        # Do the last with avg pooling to avoid checkerboard effects    
        while self.n_pixels_in // scale_factor > 4:
            conv_layers.append(tch.nn.AvgPool2d(2))
            scale_factor *= 2
            
        conv_layers.append(tch.nn.Flatten())
        self.convnet = tch.nn.Sequential(*conv_layers)
        cnn_out_size = (self.n_pixels_in // scale_factor) ** 2 * self.cnn_n_featuremaps[-1]
            
        # Build the fully connected block from cnn flat to rnn input
        fc_layers = []
        i = 0
        for n_in, n_out in zip([cnn_out_size] + self.fc_sizes[:-1], self.fc_sizes):
            fc_layers.append(tch.nn.Linear(n_in, n_out))
            fc_layers.append(tch.nn.ReLU())
            i += 1
        self.fc = tch.nn.Sequential(*fc_layers)

        if self.recurrence_type == 'rnn':
            self.recurrent_cell = tch.nn.RNNCell(self.fc_sizes[-1], self.rnn_size)
        elif self.recurrence_type == 'gru':
            self.recurrent_cell = tch.nn.GRUCell(self.fc_sizes[-1], self.rnn_size)
        elif self.recurrence_type == 'lstm':
            self.recurrent_cell = tch.nn.LSTMCell(self.fc_sizes[-1], self.rnn_size)

        # Build the second fully connected block
        self.out_layer = tch.nn.Linear(self.rnn_size, 3)
        self.device = tch.device('cuda')
        self.to(self.device)
        logging.critical(f"Model summary:\n {self}")


    def forward(self, x):
        conv_out = self.convnet(x)
        fc_out = self.fc(conv_out)

        if self.recurrence_type == 'lstm':
            hidden = (tch.nn.functional.relu(tch.zeros(x.shape[0], self.rnn_size, device=self.device)),
                      tch.nn.functional.relu(tch.zeros(x.shape[0], self.rnn_size, device=self.device)))
        else:
            hidden = tch.nn.functional.relu(tch.zeros(x.shape[0], self.rnn_size, device=self.device))

        for _ in range(self.recurrent_steps):
            hidden = self.recurrent_cell(fc_out, hidden)

        if self.recurrence_type == 'lstm':
            hidden = hidden[0]

        return self.out_layer(hidden)

if __name__ == '__main__':
    net_params = {
        'n_pixels_in': 128,
        'cnn_n_featuremaps': [128, 128, 128, 128, 128],
        'cnn_kernel_sizes': [5, 5, 3, 3, 3],
        'cnn_kernel_strides': [1, 1, 1, 1, 1],
        'fc_sizes': [1024, 512],
        'rnn_size': 512,
        'recurrence_type': 'rnn',
        'recurrent_steps': 5,
    }

    dummy_data = tch.zeros((8, 3, 128, 128)).to(tch.device('cuda'))
    net = ConvolutionalRecurrentNetwork(net_params)
    net(dummy_data)

    net_params['recurrence_type'] = 'gru'
    net = ConvolutionalRecurrentNetwork(net_params)
    net(dummy_data)

    # Check it works for smaller input from fovea
    dummy_data = tch.zeros((8, 3, 32, 32)).to(tch.device('cuda'))
    net_params['recurrence_type'] = 'lstm'
    net_params['n_pixels_in'] = 32
    net_params['cnn_n_featuremaps'] = [32, 32, 32]
    net_params['cnn_kernel_sizes'] = [3, 3, 3]
    net_params['cnn_kernel_strides'] = [1, 1, 1]
    net_params['fc_sizes'] = [256, 128]
    net_params['rnn_size'] = 128
    net = ConvolutionalRecurrentNetwork(net_params)
    net(dummy_data)

    net_params = {
                'n_pixels_in': 32,
                'cnn_n_featuremaps': [128, 64, 64],
                'cnn_kernel_sizes': [5, 5, 3,],
                'cnn_kernel_strides': [1, 1, 1,],
                'fc_sizes': [512, 512],
                'rnn_size': 512,
                'recurrence_type': 'rnn',
                'recurrent_steps': 1, # 1 step is equivalent to rec layer being feedforward, with same number of parameters !
                }
    net = ConvolutionalRecurrentNetwork(net_params)
    net(dummy_data)