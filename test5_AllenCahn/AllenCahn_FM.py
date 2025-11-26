# %% [markdown]
# # Imports

# %%
import pickle
import os
import urllib
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor

import mlflow


import matplotlib
matplotlib.use('Agg')

data_folder = r'path/to/data/folder'  # TODO: set this to your local data folder

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mlflowdir", type=str) # location of mlruns directory
args, unknown = parser.parse_known_args()

if args.mlflowdir is not None:
    abs_path = os.path.abspath(args.mlflowdir)
    mlflow.set_tracking_uri('file://' + os.path.join(abs_path, 'mlruns'))


# %%
mlflow.set_experiment('AllenCahn_FM')
print('set mlflow experiment to AllenCahn_FM')

mlflow.start_run(run_name=f'CNN_SM3_improvedUpscaling', description=f'AllenCahn_FM CNN, 3-way symmetric matching, improved the upscaling to respect the periodicity without drift')


# %%
# Save this script itself
file_path = os.path.abspath(__file__)

print(file_path)
mlflow.log_artifact(file_path)
print('Logged script')

# %%
art_path = mlflow.get_artifact_uri()
art_path = art_path[8:]  # remove 'file:///' from path
# turn %20 into spaces
art_path = urllib.parse.unquote(art_path, encoding=None, errors=None)

# if there is no drive letter, prepend another slash (e.g. for linux)
if os.path.splitdrive(art_path)[0] == '':
    art_path = '/' + art_path

print('art_path:', art_path)

# %% [markdown]
# # Open dataset

# %%
rng = np.random.default_rng(42)

# %%
dataset = 'AllenCahn_data_periodic_2000_2.pkl'
mlflow.log_param('dataset', dataset)
data_path = os.path.join(data_folder, dataset)
mlflow.log_param('data_path', data_path)
with open(data_path, 'rb') as f:
    data = pickle.load(f)

N_samples = len(data['solutions'])
inds = rng.choice(N_samples, size=3*N_samples//4, replace=False)
bools = np.zeros(N_samples, dtype=bool)
bools[inds] = True
y_train = data['solutions'][bools]
y_test = data['solutions'][~bools]

x = torch.stack((data['epsilon'], data['mu']), axis=1)
x_train = x[bools]
x_test = x[~bools]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %%
subsample = 10
mlflow.log_param('subsample', subsample)

if subsample is not None:
    y_train = y_train[:, ::subsample]
    y_test = y_test[:, ::subsample]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



# %%
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=16, shuffle=False)


# %% [markdown]
# # Choose device

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda available')
else:
    device = torch.device('cpu')
    print('cuda not available')

mlflow.log_param('device', device)


# %% [markdown]
# # Define model

# %% [markdown]
# ## Test circular padding and interpolation

# %%
arr = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
print(arr)
arr = F.pad(arr, (0, 1, 0, 0), mode='circular')
print(arr)
arr2 = F.interpolate(arr, size=(17), mode= 'linear', align_corners=True).flatten()[..., :16]
print(arr2.reshape(4,4))

# %%
plt.plot(arr2)
plt.scatter(np.arange(4)*4, arr.flatten()[:4])

# %% [markdown]
# ## Define UNet

# %%
class UNet2D_halfperiodic(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, num_layers=4, kernel_size=3, resize_factor=2):
        super(UNet2D_halfperiodic, self).__init__()
        self.depth = num_layers

        if isinstance(resize_factor, int):
            self.resize_factor = [resize_factor]*num_layers
        else:
            assert len(resize_factor) == num_layers, "resize_factor must be an int or a list of length num_layers"
            self.resize_factor = resize_factor

        # Cheatsheet:
        # Input size = [batch, in_channels, height=time steps, width=space points]

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for i in range(num_layers):
            self.encoders.append(
                nn.Sequential(
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
                    nn.Conv2d(prev_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
                    nn.ReLU(),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
                    nn.ReLU(),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
                    nn.ReLU(),
                )
            )
            self.pools.append(nn.MaxPool2d(kernel_size=self.resize_factor[i], stride=self.resize_factor[i]))
            prev_channels = hidden_channels

        self.middle_conv = nn.Sequential(
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
            nn.ReLU(),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
            nn.ReLU(),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
            nn.ReLU(),
        )

        # Decoder path
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            # After concatenation, channels = hidden_channels * 2 except at final
            in_ch = hidden_channels * 2  # if i < num_layers - 1 else hidden_channels
            self.decoders.append(
                nn.Sequential(
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
                    nn.Conv2d(in_ch, hidden_channels, kernel_size=kernel_size, padding='valid'),
                    nn.ReLU(),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
                    nn.ReLU(),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
                    nn.ReLU(),
                )
            )

        self.extra_conv = nn.Sequential(
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
            nn.ReLU(),
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ConstantPad2d((0, 0, 1, 1), 0.0),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='valid'),
            nn.ReLU(),
        )

        # Final mapping
        self.final_convs = nn.Sequential(
            nn.Conv2d(hidden_channels + in_channels, hidden_channels, kernel_size=1, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding='valid'),
        )

    def forward(self, tau, x, verbose=False):
        # Encoder
        x_in = x.clone()
        skips = []
        if verbose:
            print(f'{"Input shape         :":<30} {x.shape}')
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            if verbose:
                print(f'{"Skip shape          :":<30} {x.shape}')

            x = pool(x)

            if verbose:
                print(f'{"Encoder output shape:":<30} {x.shape}')

        x = self.middle_conv(x)

        if verbose:
            print(f'{"Middle conv shape   :":<30} {x.shape}')

        # Bottleneck output
        # Decoder
        for i in range(0, len(self.decoders)):
            # for periodic interpolation, first add periodic padding left and right, then interpolate to needed size + 4, then remove 2 extra element from left and right
            x = F.pad(x, (1, 1, 0, 0), mode='circular')
            target_size = [skips[-1].shape[2], skips[-1].shape[3]+4]
            x = F.interpolate(x, size=target_size, mode='bilinear')[..., 2:-2]
            if verbose:
                print(f'{"Decoder interpolated shape :":<30} {x.shape}')

            # Skip connection: concatenate with corresponding encoder output
            skip = skips.pop()

            x = torch.cat([x, skip], dim=1)  # if i < len(self.decoders) - 1 else x
            x = self.decoders[i](x)
            if verbose:
                print(f'{"Decoder output shape :":<30} {x.shape}')

        x = self.extra_conv(x)
        x = torch.cat((x, x_in), dim=1)  # concatenate with input for final mapping
        x = self.final_convs(x)  # final mapping to output channels
        return x

# %% [markdown]
# ## Check UNet (inc. periodicity check)

# %%
# test

UNet = UNet2D_halfperiodic(
    in_channels=15,
    out_channels=1,
    ).to(device)

temptemp = torch.rand(13, 15, 93, 123).to(device)
print(UNet(0, temptemp, verbose=True).shape)

print('\n\n')

temptemp = torch.rand(7, 15, 200, 200).to(device)
print(UNet(0, temptemp, verbose=True).shape)

print('\n\n')

temptemp = y_train[:10].to(device).unsqueeze(1)
UNet = UNet2D_halfperiodic(
    in_channels=1,
    out_channels=1,
    num_layers=5,
    ).to(device)

out = UNet(0, temptemp, verbose=True)
print(out.shape)

# %%
# # Check periodicity
# %matplotlib qt
# plt.plot(np.tile(temptemp[4, 0, -1].cpu().detach().numpy(), 2), label='input')
# plt.plot(np.tile(out[4, 0, -1].cpu().detach().numpy(), 2), label='output')
# plt.legend()
# plt.show()

# %% [markdown]
# ## Define Flow

# %%
x_train.shape, y_train.shape

# %%
class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        super(Flow, self).__init__()
        self.Unet = UNet2D_halfperiodic(
            in_channels=4,
            out_channels=1,
        )

    def forward(self, t, x_t, cond) -> Tensor:
        # t.shape = [batch size, 1]
        # x_t.shape = [batch size, 1, time steps, # points]
        # cond.shape = [batch size, 2]

        # print('forward: t.shape', t.shape)
        # print('forward: x_t.shape', x_t.shape)
        # print('forward: cond.shape', cond.shape)

        inp = torch.cat((
            x_t.clone(),
            cond.reshape(-1,2,1,1).expand(-1, 2, x_t.shape[2], x_t.shape[3]),
            t.reshape(-1,1,1,1).expand(-1, 1, x_t.shape[2], x_t.shape[3])
        ), axis=1)

        # print('forward: inp.shape', inp.shape)

        return self.Unet(t, inp, verbose=False)

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, cond: Tensor) -> Tensor:
        # t: float, current time
        # t_end: float, end time
        # x_t: shape [batch size, 1, time steps, # points], current interpolation
        # cond: shape [batch size, 2], conditioning in the form of mu and epsilon
        t_start = t_start.view(1, 1, 1, 1).expand(x_t.shape[0], 1, 1, 1)
        t_end = t_end.view(1, 1, 1, 1).expand(x_t.shape[0], 1, 1, 1)

        # print('step: x_t.shape', x_t.shape)
        # print('step: t_start.shape', t_start.shape)
        # print('step: t_end.shape', t_end.shape)
        # print('step: cond.shape', cond.shape)

        return (x_t + (t_end - t_start)
                * self(
                    t=t_start + (t_end - t_start) / 2,
                    x_t= x_t + self(x_t=x_t, t=t_start, cond=cond) * (t_end - t_start) / 2,
                    cond=cond
                       )
                )

# %%
flow = Flow().to(device)

n_params = sum(p.numel() for p in flow.parameters())
print('Total nr of parameters:', n_params)

mlflow.log_param('n_params', n_params)

# %%
target_mean = torch.mean(y_train, dim=(0, 2))
target_std = torch.std(y_train, dim=(0, 2))
print('target_mean:', target_mean[::10])
print('target_std:', target_std[::10])

target_std = target_std.reshape(1, 1, -1, 1)

# %% [markdown]
# # Define optimal circular shift

# %%
# scipy.optimize.linear_sum_assignment
# to do: use python OT library? https://pythonot.github.io/index.html
# also interesting: scipy.spatial.transform.Rotation.align_vectors
# scipy.optimize.quadratic_assignment (graph matching)

# %%
import scipy.optimize as scopt

# %%
import numpy as np

def optimal_circular_shift_nd(a, b, return_corr=False):
    """
    Finds the optimal circular shift along the last axis of N-dimensional array b that maximizes the cross-correlation between a and b.

    Parameters:
    a, b : np.ndarray, shape [..., T, N]
        Arrays with the same shape, with at least 2 dimensions. T is the time dimension, and N is the space dimension.

    Returns:
    optimal_shift : np.ndarray of ints, shape [...]
        For each b[i], the shift along the last axis that maximizes similarity.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Compute FFT along the last axis
    fft_a = np.fft.fft(a, axis=-1) # still shape [..., T, N]
    fft_b = np.fft.fft(b, axis=-1)

    # Compute cross-correlation using inverse FFT
    corr = np.fft.ifft(fft_a * np.conj(fft_b), axis=-1).real # still shape [..., T, N]

    # Aggregate across time steps
    total_corr = np.sum(corr, axis=-2)  # shape [..., N]
    optimal_shift = np.argmax(total_corr, axis=-1) # shape [...]

    if return_corr:
        optimal_corr = np.take_along_axis(total_corr, optimal_shift[..., np.newaxis], axis=-1)[..., 0]
        return optimal_shift, optimal_corr
    return optimal_shift

# Example usage with 2D arrays
a = np.array([[[[1.0, 4.3, 2.2, 9.1], [ 1.0, 4.3, 2.2, 9.1]],
              [[2.2, 9.1, 1.0, 4.3], [ 2.2, 9.1, 1.0, 4.3]]]]) # [1,2, 2, 4]
b = np.array([[[[4.0, 2.1, 8.9, 1.2], [4.0, 2.1, 8.9, 1.2]],
              [[1.0, 4.3, 2.2, 9.1], [ 1.0, 4.3, 2.2, 9.1]]]])  # [1,2, 2, 4]
shift = optimal_circular_shift_nd(a, b)

print("Optimal shift:", shift)

# %%
import numpy as np

def optimal_circular_shift_nd_all_combinations(a, b):
    """
    For each pairing (a[i], b[j]), finds the optimal circular shift of b along the last axis of N-dimensional arrays a and b that maximizes the cross-correlation between them.

    Parameters:
    a : np.ndarray, shape [B1, T, N]
    b : np.ndarray, shape [B2, T, N]
        B1 and B2 are the batch sizes, T is the time dimension, and N is the space dimension

    Returns:
    optimal_shift : np.ndarray, shape [B1, B2]
        The shift along the last axis of b that maximizes similarity.
    optimal_corr : np.ndarray, shape [B1, B2]
        The maximum correlation values for each pairing.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Compute FFT along the last axis
    fft_a = np.fft.fft(a, axis=-1)
    fft_b = np.fft.fft(b, axis=-1)

    # Compute cross-correlation using inverse FFT
    corr = np.fft.ifft(fft_a[:, np.newaxis, :] * np.conj(fft_b)[np.newaxis], axis=-1).real  # shape [B1, B2, T, N]

    # Aggregate across time steps
    total_corr = np.sum(corr, axis=-2)  # shape (B1, B2, N)
    optimal_shift = np.argmax(total_corr, axis=-1) # shape (B1, B2)
    optimal_corr = np.max(total_corr, axis=-1)  # shape (B1, B2)

    return optimal_shift, optimal_corr

# Example usage with 2D arrays
a = np.array([[[1.0, 4.3, 2.2, 9.1], [ 1.0, 4.3, 2.2, 9.1]],
              [[2.2, 9.1, 1.0, 4.3], [ 2.2, 9.1, 1.0, 4.3]]]) # [2, 2, 4]
b = np.array([[[4.0, 2.1, 8.9, 1.2], [4.0, 2.1, 8.9, 1.2]],
              [[1.0, 4.3, 2.2, 9.1], [ 1.0, 4.3, 2.2, 9.1]]])  # [2, 2, 4]
shift, corr = optimal_circular_shift_nd_all_combinations(a, b)

print("Optimal shift:", shift)
print("Maximum correlation:", corr)

# %%
# Goal: c_tilde [bs, bs], optimal_shift [bs, bs]
# where c_tilde[i,j] = optimal_circular_shift(x_0[i], x_1[j])[0]
# input: x_0 [bs, T, N]
# input: x_1 [bs, T, N]

# %%
def roll_batched(arr, shift):
    """
    Rolls the array along the last axis, by the given shift which differs for each batch.

    Parameters:
    arr : np.ndarray [B, T, N]
        The input array to roll.
    shift : np.ndarray, shape [B]
        The number of positions to shift the array. Positive values roll to the right, and negative values roll to the left.

    Returns:
    np.ndarray : shape = arr.shape
        The rolled array.
    """
    # Ensure the shift is within the bounds of the array size along the specified axis
    shift = np.asarray(shift)
    arr = np.asarray(arr)

    shift = shift % arr.shape[-1]

    rows, column_indices = np.ogrid[:arr.shape[0], :arr.shape[-1]]
    column_indices = column_indices - shift[:, np.newaxis]

    result = arr[rows, :, column_indices]
    return np.transpose(result, axes=(0,2,1))


# Example:
arr = np.array([
    [[1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]],

    [[10, 20, 30, 40, 50],
    [60, 70, 80, 90, 100]]
])
arr_out = np.array([
    [[5, 1, 2, 3, 4],
    [10, 6, 7, 8, 9]],

    [[30, 40, 50, 10, 20],
    [80, 90, 100, 60, 70]]
])

shift = np.array([1, -2])
rolled = roll_batched(arr, shift)

print("Original array:")
print(arr)
print("Rolled array:")
print(rolled)

# %%
# Test optimal shift
plt.figure()
plt.imshow(y_train[0].T, vmin=-1, vmax=1, aspect='auto')
# plt.show()
plt.close()

y_train_rolled_perturbed = roll_batched(y_train[[0]], [50]) + 0.1 * np.random.randn(*y_train[[0]].shape)

print(optimal_circular_shift_nd(y_train_rolled_perturbed, y_train[[0]]))

plt.figure()
plt.imshow(y_train_rolled_perturbed.T, vmin=-1, vmax=1, aspect='auto')
# plt.show()
plt.close()

# %%
# Test roll_batched
plt.figure()
plt.imshow(y_train[0].T, vmin=-1, vmax=1, aspect='auto')
# plt.show()
plt.close()

plt.figure()
plt.imshow(roll_batched(y_train[[0]], [50]).T, vmin=-1, vmax=1, aspect='auto')
# plt.show()
plt.close()

# %%
# Create reflected and inverted versions of the original batch of trajectories
def reflect_and_invert(trajectories):
    """Reflect and invert the given batch of trajectories.

    Parameters
    ----------
    trajectories : ndarray of shape [B, T, N]
        The input batch of trajectories. B is the batch size, T is the number of time steps, and N is the number of spatial points.

    Returns
    -------
    transformed trajectories : ndarray of shape [B, 4, T, N]
        The transformed trajectories, including the original, reflected, inverted and reflected+inverted versions.
    """
    B, T, N = trajectories.shape
    transformed = np.zeros((B, 4, T, N))

    # Original
    transformed[:, 0] = trajectories

    # Reflected
    transformed[:, 1] = np.flip(trajectories, axis=-1) # flip order of spatial points to reflect

    # Inverted
    transformed[:, 2] = -trajectories  # Invert y-values

    # Reflected + Inverted
    transformed[:, 3] = -np.flip(trajectories, axis=-1)  # Reflect and invert

    return transformed

# %%
# Test reflect_and_invert
for i in range(2):
    plt.figure()
    plt.imshow(y_train[i].T, vmin=-1, vmax=1, aspect='auto')
    plt.title(f'Original, case {i}')
    # plt.show()
    plt.close()

transf = reflect_and_invert(y_train[:2].numpy())

for i in range(2):
    for j in range(4):
        plt.figure()
        plt.imshow(transf[i, j].T, vmin=-1, vmax=1, aspect='auto')
        plt.title(f'Transformed, case {i}, variant {j}')
        # plt.show()
        plt.close()

# %% [markdown]
# # Train

# %%
from scipy.ndimage import gaussian_filter

# %%
symmetric_matching = True
do_reflect_and_invert = True # extra symmetric matching

apply_gaussian_filter = False # different prior: correlate nearby spatial points
sigma = 2  # sigma for the gaussian filter

mlflow.log_param('apply_gaussian_filter', apply_gaussian_filter)
mlflow.log_param('symmetric_matching', symmetric_matching)
mlflow.log_param('do_reflect_and_invert', do_reflect_and_invert)
mlflow.log_param('sigma', sigma)

loss_fn = nn.MSELoss()

training_schedule = [[1e-3, 1e-4, 1e-5], [200, 200, 100]]
n_epochs = np.sum(training_schedule[1])
mlflow.log_param('training_schedule', training_schedule)
mlflow.log_param('n_epochs', n_epochs)

cum_eps = np.cumsum(training_schedule[1])
lr_ind = 0
optimizer = torch.optim.Adam(flow.parameters(), training_schedule[0][lr_ind])

MSE_loss = []
for i in range(n_epochs):

    loss_temp = []
    for j, [cond, target] in enumerate(train_loader):
        T = target.shape[1]
        N = target.shape[2]
        x_1 = target.unsqueeze(1)  # shape [batch size, 1, T, N]

        # Initial guess
        x_0 = torch.randn_like(x_1) # initial Gaussian noise
        x_0 = torch.cumsum(x_0, dim=2)  # cumulative sum over time
        if apply_gaussian_filter:
            x_0 = torch.as_tensor(gaussian_filter(x_0.cpu().numpy(), sigma=sigma, mode='wrap', axes=-1), dtype=torch.float32)  # apply Gaussian filter to make neighboring points more similar
        mean_disp = 1.0*torch.sqrt(2*(torch.arange(T, dtype=torch.float32)+1)/torch.pi)  # 𝜎√(2𝑁/𝜋)  # mean random walk displacement for each time step
        x_0 = x_0 / mean_disp.unsqueeze(-1) * target_std  # scale to match target std

        # for the first six data points of the first batch of the first epoch, plot the before and after of symmetric matching to see if it makes sense
        if j == 0 and i == 0:
            for b in range(6):  # range(len(cond)):
                plt.figure(figsize=(3,2))
                plt.imshow(x_1.squeeze(1)[b].T.cpu().detach().numpy(), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                plt.title(f'Target {b}')
                plt.colorbar()
                # plt.show()
                plt.close()

        if symmetric_matching:

            if do_reflect_and_invert:

                x_0_temp = x_0.expand(-1, 4, -1, -1).cpu().numpy() # shape [B, 4, T, N]
                x_1 = reflect_and_invert(x_1.squeeze(1).cpu().numpy()) # shape [B, 4, T, N]

                shift, corr = optimal_circular_shift_nd(x_0_temp, x_1, return_corr=True)  # both shape [B, 4]
                ind = np.argmax(corr, axis=-1, keepdims=True)  # shape [B, 1], which of the 4 variants of x_1 is best for each pair
                x_1 = np.take_along_axis(x_1, ind[..., np.newaxis, np.newaxis], axis=1)[:, 0]  # shape [B, T, N]
                shift = np.take_along_axis(shift, ind, axis=1).flatten()

                # roll x_1 according to optimal shifts
                x_1 = torch.tensor(
                    roll_batched(x_1, shift), dtype=torch.float32
                    ).unsqueeze(1).to(device)

            else:
                shift = optimal_circular_shift_nd(
                    x_0.squeeze(1).cpu().numpy(),
                    x_1.squeeze(1).cpu().numpy())

                # roll x_1 according to optimal shifts
                x_1 = torch.from_numpy(
                        roll_batched(x_1.squeeze(1).cpu().numpy(), shift)
                    ).unsqueeze(1).to(device)

        if j == 0 and i == 0:
            for b in range(6):  # range(len(cond)):
                plt.figure(figsize=(3,2))
                plt.imshow(x_0.squeeze(1)[b].T.cpu().detach().numpy(), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                plt.title(f'Initial Guess {b}')
                plt.colorbar()
                # plt.show()
                plt.close()


                plt.figure(figsize=(3,2))
                plt.imshow(x_1.squeeze(1)[b].T.cpu().detach().numpy(), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                plt.title(f'Target {b} after symmetric matching')
                plt.colorbar()
                # plt.show()
                plt.close()

        x_0 = x_0.to(device)
        x_1 = x_1.to(device)

        cond = cond.to(device)  # things to condition on

        t = torch.rand(x_0.shape[0], 1, 1, 1).to(device)

        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        optimizer.zero_grad()

        dx_t_pred = flow(t=t, x_t=x_t, cond=cond)
        # print('dx_t_pred.shape', dx_t_pred.shape)
        # print('dx_t.shape', dx_t.shape)
        loss = loss_fn(dx_t_pred, dx_t)
        loss.backward()
        optimizer.step()

        loss_temp.append(loss.item())

    # increase learning rate according to schedule
    if i == cum_eps[lr_ind]:
        lr_ind += 1
        print(f'Epoch {i}, switching to learning rate {training_schedule[0][lr_ind]}')
        optimizer = torch.optim.Adam(flow.parameters(), training_schedule[0][lr_ind])

    print(f'{i:<4} Loss: {np.mean(loss_temp):.5f}')
    MSE_loss.append(np.mean(loss_temp))
    mlflow.log_metric('loss', np.mean(loss_temp), step=i)

# %%
plt.figure()
plt.plot(MSE_loss)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
mlflow.log_figure(plt.gcf(), 'Loss_curve.png')
# plt.show()
plt.close()

# %%
# Save network parameters
# torch.save(flow.state_dict(), 'allen_cahn_model8...?.pt')

# %%
# save parameters of model
try:
    path = os.path.join(art_path, 'model_weights.pt')
    torch.save(flow.state_dict(), path)
    print('Saved model parameters to', path)
except AttributeError as e:
    print('Error saving model parameters:', repr(e))

# Save entire model
try:
    path = os.path.join(art_path, 'model.pt')
    torch.save(flow, path)
    print('Saved model to', path)
except AttributeError as e:
    print('Error saving model:', repr(e))
except pickle.PicklingError as e:
    print('Error saving model:', repr(e))


# %% [markdown]
# # Test flow

# %%
# %matplotlib qt
n_steps = 32
n_plots = 5

time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)

inds_to_plot = np.arange(n_steps)[(np.arange(n_plots)/(n_plots-1)*(n_steps-1)).astype(int)]

ind = 11 #0 #3
with torch.no_grad():
    for j, [epsmu, target] in enumerate(test_loader):
        cond = epsmu.to(device)  # things to condition on

        x_1 = target.unsqueeze(1).to(device)

        for ind in [0, 3, 11]:
            # Initial guess
            # set random seed
            torch.manual_seed(42)

            x = torch.randn_like(x_1, device=device) # initial Gaussian noise
            x = torch.cumsum(x, dim=2)  # cumulative sum over time
            # mean random walk displacement for each time
            if apply_gaussian_filter:
                x = torch.as_tensor(gaussian_filter(x.cpu().numpy(), sigma=sigma, mode='wrap', axes=-1), dtype=torch.float32).to(device)  # apply Gaussian filter to make neighboring points more similar
            mean_disp = 1.0*torch.sqrt(2*(torch.arange(T, dtype=torch.float32, device=device)+1)/torch.pi)  # 𝜎√(2𝑁/𝜋)
            x = x / mean_disp.unsqueeze(-1) * target_std.to(device)  # scale to match target std

            pred = x.cpu().detach()

            fig, axes = plt.subplots(1, n_plots+1, figsize=(20, 3), sharex=True, sharey=True)
            fig.suptitle(f'{n_steps} flow matching steps')

            for ax in axes:
                ax.set_aspect('equal')
                ax.set_xlabel('Time')
                ax.set_ylabel('Space')
                # ax.axline((0,0), (1,1), c='tab:red')
                # ax.axline((0,0), (-1,1), c='tab:red')

            axes[0].imshow(pred[ind, 0].T.cpu().detach().numpy(), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1, extent=[0, x.shape[2], 0, 1])
            axes[0].set_title(f't = {time_steps[0]:.2f}')

            plot_ind = 1
            for i in range(n_steps):
                # print(f'\nStep {i}')
                # print(x.shape)
                x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], cond=cond)

                if i == inds_to_plot[plot_ind]:
                    pred = x.cpu().detach()
                    axes[plot_ind].imshow(pred[ind, 0].T.cpu().detach().numpy(), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1, extent=[0, x.shape[2], 0, 1])
                    axes[plot_ind].set_title(f't = {time_steps[i + 1]:.2f}')

                    plot_ind += 1

            axes[-1].imshow(target[ind].T.cpu().detach().numpy(), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1, extent=[0, x.shape[2], 0, 1])
            axes[-1].set_title(f'Ground Truth')

            plt.tight_layout()

            mlflow.log_figure(plt.gcf(), f"test_{ind}.png")
            # plt.show()
            plt.close()

        break


# %% [markdown]
# # Vary $\mu$

# %%
# %matplotlib qt
n_steps = 32

time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
epsilon = 0.1

avg_values = []
with torch.no_grad():
    # set random seed
    torch.manual_seed(42)
    N_samples = 500
    mu_arr = torch.linspace(-0.1, 1.0, N_samples, dtype=torch.float32).to(device)
    eps_arr = torch.tensor([0.1]*N_samples, dtype=torch.float32).to(device)
    cond = torch.cat([eps_arr.unsqueeze(1), mu_arr.unsqueeze(1)], dim=1)

    test_loader2 = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(cond),
        batch_size=32,
        shuffle=False
    )
    for j, cond_temp in enumerate(test_loader2):

        cond_temp = cond_temp[0].to(device)

        # Initial guess
        x = torch.randn(len(cond_temp), 1, T, N, device=device) # initial Gaussian noise
        x = torch.cumsum(x, dim=2)  # cumulative sum over time
        # mean random walk displacement for each time
        if apply_gaussian_filter:
            x = torch.as_tensor(gaussian_filter(x.cpu().numpy(), sigma=sigma, mode='wrap', axes=-1), dtype=torch.float32).to(device)  # apply Gaussian filter to make neighboring points more similar
        mean_disp = 1.0*torch.sqrt(2*(torch.arange(T, dtype=torch.float32, device=device)+1)/torch.pi)  # 𝜎√(2𝑁/𝜋)
        x = x / mean_disp.unsqueeze(-1) * target_std.to(device)  # scale to match target std

        for i in range(n_steps):
            print(f'\nStep {i}')
            print(x.shape)
            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], cond=cond_temp)

        avg_values.extend(x[..., 0, -1, :].mean(dim=-1).cpu().numpy())

# %%
plt.figure(figsize=(4,3), dpi=200)
plt.scatter(mu_arr.cpu().numpy(), avg_values, label='Predicted', s=3)
plt.xlabel(r'$\mu$')
plt.ylabel(r'Average solution value')
mlflow.log_figure(plt.gcf(), "mu_vs_avg_solution.png")
# plt.subplots_adjust(left=0.2, bottom=0.2)

# %% [markdown]
# # Vary $\epsilon$

# %%
# %matplotlib qt
n_steps = 32

time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
epsilon = 0.1

sol_norm = []
with torch.no_grad():
    # set random seed
    torch.manual_seed(42)
    N_samples = 500
    eps_arr = 10**torch.linspace(-3, -0.5, N_samples, dtype=torch.float32).to(device)
    mu_arr = torch.tensor([1.0]*N_samples, dtype=torch.float32).to(device)
    cond = torch.cat([eps_arr.unsqueeze(1), mu_arr.unsqueeze(1)], dim=1)

    test_loader2 = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(cond),
        batch_size=32,
        shuffle=False
    )
    for j, cond_temp in enumerate(test_loader2):

        cond_temp = cond_temp[0].to(device)

        # Initial guess
        x = torch.randn(len(cond_temp), 1, T, N, device=device) # initial Gaussian noise
        x = torch.cumsum(x, dim=2)  # cumulative sum over time
        # mean random walk displacement for each time
        if apply_gaussian_filter:
            x = torch.as_tensor(gaussian_filter(x.cpu().numpy(), sigma=sigma, mode='wrap', axes=-1), dtype=torch.float32).to(device)  # apply Gaussian filter to make neighboring points more similar
        mean_disp = 1.0*torch.sqrt(2*(torch.arange(T, dtype=torch.float32, device=device)+1)/torch.pi)  # 𝜎√(2𝑁/𝜋)
        x = x / mean_disp.unsqueeze(-1) * target_std.to(device)  # scale to match target std

        for i in range(n_steps):
            print(f'\nStep {i}')
            print(x.shape)
            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], cond=cond_temp)

        sol_norm.extend(torch.norm(x[..., 0, -1, :], dim=-1).cpu().numpy())

# %%
# plt.figure(figsize=(4,3), dpi=200)
plt.figure()
plt.scatter(eps_arr.cpu().numpy(), sol_norm, label='Predicted', s=3)
plt.xlabel('$\epsilon$')
plt.ylabel('Solution norm')
# plt.xscale('log')
mlflow.log_figure(plt.gcf(), "epsilon_vs_solution_norm.png")
# plt.subplots_adjust(left=0.2, bottom=0.2)

# %% [markdown]
# # Plot ground truth

# %%
# with torch.no_grad():
#     for j, [epsmu, target] in enumerate(test_loader):
#         if j < 3: continue
#         for i in range(len(target)):
#             plt.figure()
#             plt.imshow(target[i].T.cpu().detach().numpy(), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1, extent=[0, target.shape[2], 0, 1])
#             plt.title(f'Ground Truth, $\\epsilon={epsmu[i, 0]:.2f}, \\mu={epsmu[i, 1]:.2f}$')
#             plt.colorbar()
#             plt.show()

#         if j == 4:
#             break

# %% [markdown]
# # Count nr of blobs

# %%
arr = torch.randn((3, 15))/5
print(arr)
arr_temp = torch.sign(arr)
print(arr_temp)
arr_temp[torch.abs(arr) < 0.1] = 0.0
print(arr_temp)

# Replace zeros with NaN to ignore them
arr_no_zeros = arr_temp.clone()
arr_no_zeros[arr_no_zeros == 0] = float('nan')
print(arr_no_zeros)

# Forward fill NaNs along each row
mask = ~torch.isnan(arr_no_zeros)
print('mask:\n', mask, sep='')
idx = torch.arange(arr_temp.size(1)).repeat(arr_temp.size(0), 1)
idx[~mask] = 0
print('idx:\n', idx, sep='')
idx = torch.cummax(idx, dim=1)[0] # get the indices of the last non-NaN value or zero if all up to that position are NaN
print('idx:\n', idx, sep='')
filled = torch.gather(arr_no_zeros, 1, idx)
print('filled:\n', filled, sep='')

# Compare signs of adjacent elements
sign_changes = (torch.sign(filled[:, 1:]) != torch.sign(filled[:, :-1])) & \
                ~torch.isnan(filled[:, 1:]) & ~torch.isnan(filled[:, :-1])

print('sign_changes:\n', sign_changes, sep='')

print('sign_changes.sum(dim=1):\n', sign_changes.sum(dim=1), sep='')

# %%
def count_sign_switches(arr, cutoff=0.1):
    arr_temp = torch.sign(arr)
    arr_temp[torch.abs(arr) < cutoff] = 0.0

    # Replace zeros with NaN to ignore them
    arr_no_zeros = arr_temp.clone()
    arr_no_zeros[arr_no_zeros == 0] = float('nan')

    # Forward fill NaNs along each row
    mask = ~torch.isnan(arr_no_zeros)
    idx = torch.arange(arr_temp.size(1), device=arr.device).repeat(arr_temp.size(0), 1)
    idx[~mask] = 0
    idx = torch.cummax(idx, dim=1)[0]
    filled = torch.gather(arr_no_zeros, 1, idx)

    # Compare signs of adjacent elements
    sign_changes = (torch.sign(filled[:, 1:]) != torch.sign(filled[:, :-1])) & \
                   ~torch.isnan(filled[:, 1:]) & ~torch.isnan(filled[:, :-1])

    # Count sign changes per row
    return sign_changes.sum(dim=1)

# %%
arr = torch.tensor([[0.05, -0.05, 0.09, -0.08, 0.02, -0.01]])
count_sign_switches(arr)

# %%
arr = torch.randn((3, 4))/5
print(arr)

print(count_sign_switches(arr))

# %%
# %matplotlib qt
n_steps = 32

time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
epsilon = 0.1

n_blobs = []
n_blobs_real = []
with torch.no_grad():
    # set random seed
    torch.manual_seed(42)

    for j, [epsmu, target] in enumerate(test_loader):

        cond = epsmu.to(device)

        # Initial guess
        x = torch.randn(len(cond), 1, T, N, device=device) # initial Gaussian noise
        x = torch.cumsum(x, dim=2)  # cumulative sum over time
        if apply_gaussian_filter:
            x = torch.as_tensor(gaussian_filter(x.cpu().numpy(), sigma=sigma, mode='wrap', axes=-1), dtype=torch.float32).to(device)  # apply Gaussian filter to make neighboring points more similar
        # mean random walk displacement for each time
        mean_disp = 1.0*torch.sqrt(2*(torch.arange(T, dtype=torch.float32, device=device)+1)/torch.pi)  # 𝜎√(2𝑁/𝜋)
        x = x / mean_disp.unsqueeze(-1) * target_std.to(device)  # scale to match target std

        for i in range(n_steps):
            # print(f'\nStep {i}')
            # print(x.shape)
            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], cond=cond)

        x_temp = x[:, 0, -1]
        # print('x_temp:\n', x_temp, sep='')
        n_blobs_temp = count_sign_switches(x_temp, cutoff=0.2).cpu().numpy()
        # print('n_blobs_temp:\n', n_blobs_temp, sep='')
        n_blobs.extend(n_blobs_temp)

        targ = target[:, -1].to(device)
        n_blobs_temp_2 = count_sign_switches(targ, cutoff=0.2).cpu().numpy()
        # print('n_blobs_temp_2:\n', n_blobs_temp_2, sep='')
        n_blobs_real.extend(n_blobs_temp_2)

        if (np.abs(n_blobs_temp_2 - n_blobs_temp) > 20).any():
            inds = np.where(np.abs(n_blobs_temp_2 - n_blobs_temp) > 20)[0]
            inds = torch.tensor(inds, device=device)

            plt.figure()
            plt.suptitle(f'Output {j} {inds[0]}, {n_blobs_temp[inds[0]]} blobs')
            plt.imshow(x[inds[0], 0].T.cpu().detach().numpy(), label='output', vmin=-1, vmax=1, aspect='auto')
            # plt.show()
            plt.close()

            plt.figure()
            plt.suptitle(f'Target {j} {inds[0]}, {n_blobs_temp_2[inds[0]]} blobs')
            plt.imshow(target[inds[0]].T.cpu().detach().numpy(), label='target', vmin=-1, vmax=1, aspect='auto')
            # plt.show()
            plt.close()

            plt.figure()
            plt.suptitle(f'Comparison {j} {inds[0]}, {n_blobs_temp[inds[0]]} blobs predicted, {n_blobs_temp_2[inds[0]]} blobs target')
            plt.plot(x[inds[0], 0, -1].cpu().detach().numpy(), label='output')
            plt.plot(target[inds[0], -1].cpu().detach().numpy(), label='target')
            plt.legend()
            # plt.show()
            plt.close()

# %%
plt.figure(figsize=(3,5), dpi=200)
plt.title('Nr of blobs in last time step')
plt.scatter(n_blobs_real, n_blobs, s=5, alpha=0.1)
plt.axline((0, 0), slope=1, color='tab:red', zorder=-3)
plt.gca().set_aspect('equal')
plt.xlabel('Real')
plt.ylabel('Predicted')
mlflow.log_figure(plt.gcf(), "n_blobs.png")

# %% [markdown]
# # Investigate prior

# # %%
# arr = np.random.normal(size=(100,))
# plt.plot(arr, label='Input')
# print('std arr:', np.std(arr))
# gau_arr = gaussian_filter(arr, sigma=1, mode='wrap', axes=-1)
# plt.plot(gau_arr, label='Gaussian filtered, s=1')
# print('std gau_arr (s=1):', np.std(gau_arr))
# gau_arr = gaussian_filter(arr, sigma=2, mode='wrap', axes=-1)
# plt.plot(gau_arr, label='Gaussian filtered, s=2')
# print('std gau_arr (s=2):', np.std(gau_arr))
# plt.legend()

# %%
x = torch.randn(100, T, N, device=device) # initial Gaussian noise
x = torch.cumsum(x, dim=1)  # cumulative sum over time
x = torch.as_tensor(gaussian_filter(x.cpu().numpy(), sigma=sigma, mode='wrap', axes=-1), dtype=torch.float32).to(device)  # apply Gaussian filter to make neighboring points more similar
pred_sig = torch.sqrt(torch.arange(T, dtype=torch.float32, device=device)+1)  # √𝑁

plt.plot(pred_sig.cpu().numpy(), label='Predicted Standard Deviation')
plt.plot(np.std(x.cpu().numpy(), axis=(0,2)), label='Actual Standard Deviation')
plt.legend()

# %%
x = torch.randn(100, T, N, device=device) # initial Gaussian noise
x = torch.cumsum(x, dim=1)  # cumulative sum over time
pred_sig = torch.sqrt(torch.arange(T, dtype=torch.float32, device=device)+1)  # √𝑁

plt.plot(pred_sig.cpu().numpy(), label='Predicted Standard Deviation')
plt.plot(np.std(x.cpu().numpy(), axis=(0,2)), label='Actual Standard Deviation')
plt.legend()

# %% [markdown]
# # Calculate residual

# %%
from scipy.sparse import diags
L = 1.0
N = y_train.shape[2]  # Number of spatial points

dx = L / N        # Spatial step size
x = np.linspace(0, L, N, endpoint=False)
dt = 1.0

# Construct Laplacian with periodic boundary conditions
main_diag = -2.0 * np.ones(N)
off_diag = np.ones(N - 1)
laplacian = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], shape=(N, N)).toarray()
laplacian[0, -1] = laplacian[-1, 0] = 1.0  # periodic BC
laplacian = laplacian / dx**2

# %%
# %matplotlib qt
n_steps = 4

time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)

residual_real = []
residual_pred = []
with torch.no_grad():
    # set random seed
    torch.manual_seed(42)

    for j, [epsmu, target] in enumerate(test_loader):
        print(f'Batch {j}')

        cond = epsmu.to(device)

        # Initial guess
        x = torch.randn(len(cond), 1, T, N, device=device) # initial Gaussian noise
        x = torch.cumsum(x, dim=2)  # cumulative sum over time
        if apply_gaussian_filter:
            x = torch.as_tensor(gaussian_filter(x.cpu().numpy(), sigma=sigma, mode='wrap', axes=-1), dtype=torch.float32).to(device)  # apply Gaussian filter to make neighboring points more similar
        # mean random walk displacement for each time
        mean_disp = 1.0*torch.sqrt(2*(torch.arange(T, dtype=torch.float32, device=device)+1)/torch.pi)  # 𝜎√(2𝑁/𝜋)
        x = x / mean_disp.unsqueeze(-1) * target_std.to(device)  # scale to match target std

        for i in range(n_steps):
            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], cond=cond)

        target = target.numpy()
        x = x[:, 0].cpu().numpy()

        # Allen-Cahn equation: du/dt = eps^2 * laplacian u - (u^3 - mu*u)

        eps = cond[:, 0].reshape(-1, 1, 1).cpu().numpy()
        mu = cond[:, 1].reshape(-1, 1, 1).cpu().numpy()

        # laplacian shape [N, N]
        # x.shape: [B, T, N]
        # cond.shape: [B,]
        lhs1 = (x[:, 1:] - x[:, :-1])/dt
        rhs1a = eps**2 * np.einsum('ij,kmj->kmi', laplacian, x[:, 1:])
        rhs1b = - (x[:, :-1]**3 - mu * x[:, :-1])
        diff1 = lhs1 - (rhs1a + rhs1b)  # shape [B, T-1, N]
        residual_pred.extend(np.linalg.norm(diff1, axis=(1,2)).tolist())

        lhs2 = (target[:, 1:] - target[:, :-1])/dt
        rhs2 = eps**2 * np.einsum('ij,kmj->kmi', laplacian, target[:, 1:]) - (target[:, :-1]**3 - mu * target[:, :-1])
        diff2 = lhs2 - rhs2  # shape [B, T-1, N]
        residual_real.extend(np.linalg.norm(diff2, axis=(1,2)).tolist())

        if j == 0:
            plt.figure()
            plt.imshow(x[0].T, vmin=-1, vmax=1, aspect='auto', cmap='coolwarm')
            plt.title('Predicted')
            plt.colorbar()

            plt.figure()
            plt.imshow(rhs1a[0].T, aspect='auto', cmap='coolwarm')
            plt.title('Predicted rhs a (eps^2 * laplacian u)')
            plt.colorbar()

            plt.figure()
            plt.imshow(rhs1b[0].T, aspect='auto', cmap='coolwarm')
            plt.title('Predicted rhs b (-(u^3 - mu*u))')
            plt.colorbar()

            plt.figure()
            plt.imshow(lhs1[0].T, aspect='auto', cmap='coolwarm')
            plt.title('Predicted lhs (du/dt)')
            plt.colorbar()

            plt.figure()
            plt.imshow(target[0].T, vmin=-1, vmax=1, aspect='auto', cmap='coolwarm')
            plt.title('Target')
            plt.colorbar()

            plt.figure()
            plt.imshow(diff1[0].T, aspect='auto', cmap='coolwarm')
            plt.title('Residual in predicted')
            plt.colorbar()

            plt.figure()
            plt.imshow(diff2[0].T, aspect='auto', cmap='coolwarm')
            plt.title('Residual in target')
            plt.colorbar()

        # break

print('Residual (Real):', np.mean(residual_real))
print('Residual (Predicted):', np.mean(residual_pred))

mlflow.log_metric('residual_pred', np.mean(residual_pred))

# %%

# end mlflow run
mlflow.end_run()


