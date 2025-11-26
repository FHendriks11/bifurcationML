# %%
# To do: track more things in mlflow instead of putting it all in  the description
# Fix naming of models: e.g. GNNtimeConv -> GNN, make timeConv a parameter
# make more things a parameter: e.g., kernel size, resize factor, number of layers in UNet, nr of MP steps
# %%
import pickle
import os
import urllib
import time
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing

import mlflow

import matplotlib
matplotlib.use('Agg')

data_folder = r'path/to/data/folder'  # TODO: set this to the local data folder
mlflow.set_tracking_uri(r'file:///path/to/local/mlruns')  # TODO: set this to the local mlruns folder

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mlflowdir", type=str) # location of mlruns directory
args, unknown = parser.parse_known_args()

if args.mlflowdir is not None:
    abs_path = os.path.abspath(args.mlflowdir)
    mlflow.set_tracking_uri('file://' + os.path.join(abs_path, 'mlruns'))


# %%
torch.cuda.is_available()  # check if CUDA is available

# %%
# check cuda version
print(torch.version.cuda)

# %%
mlflow.set_experiment('BucklingBeams_FM')
print('set mlflow experiment to BucklingBeams_FM')

network_type = 'EGNNtimeConv'
print('network_type:', network_type)
mlflow.start_run(run_name=f'{network_type}', description=f"{network_type}, no scaling of K_i, fully connected graph, narrow/bimodal prior, UNet: back to old kernel_size etc, upscale using target size, no EGNN over time, train for closest, more MP steps, adjusted learning rate schedule")

mlflow.log_param('network_type', network_type)


# %% [markdown]
# # Open Dataset

# %%
data_path = os.path.join(data_folder, 'BucklingBeams_data_fullyConnected.pkl')
mlflow.log_param('data_path', data_path)
with open(data_path, 'rb') as f:
    data = pickle.load(f)

for graph in data['data_tr'][:5]:
    print(graph)

# %% [markdown]
# # Preprocessing

# %% [markdown]
# Scaling relation:
# if $d \rightarrow Ad$, $L_i \rightarrow AL_i$ and $K_i \rightarrow AK_i$  with $A$ some kind of scaling constant, then the outputs $\lambda$, $q_i$, $\epsilon_i$ stay the same, which means the node positions are also scaled with $A$, so $\vec{x}_i \rightarrow A\vec{x}_i$ if the origin is kept at the location of the first node.
#
# Therefore, we scale $K_i$ with $L_i$, such that only the distance quantities are equivariant.
#
# We also remove $L_i$ from the edge attributes (will be recalculated from the node positions).

# %%
data['data_tr'][0]

# %%
for key in ['data_tr', 'data_te']:
    for graph in data[key]:
        N = graph.N[0].item()

        # divide K_i by L_i to get proper scaling
        Li = graph.edge_attr[:N, [0]]
        # graph.node_attr[0, 3:] /= Li[0]  # divide K_i node 0 by L_0
        # graph.node_attr[1:N, 3:] /= 0.5*(Li[:N-1] + Li[1:])   # divide K_i nodes 1-N by 1/2(L_i+L_i-1)

        # move L_i feature to its own tensor (should be treated separately because of scaling)
        graph.L_init = graph.edge_attr[:, 0]
        graph.edge_attr = graph.edge_attr[:, 1:]  # remove L_i from edge_attr

        graph.L_tot = graph.L_init[:N].sum().reshape(1,) # total length of beam

# %%
n_train = len(data['data_tr'])  # 1  #
bs = 64  # batch size for training

mlflow.log_param('n_train', n_train)
mlflow.log_param('bs', bs)

train_loader = tg.loader.DataLoader(data['data_tr'][:n_train], batch_size=bs, shuffle=True)  #100000)  #64)
test_loader  = tg.loader.DataLoader(data['data_te'], batch_size=len(data['data_te'])) # all test data at once

# %%
print('One graph:')
print(data['data_tr'][0], '\n')

print('One batch:')
for batch in train_loader:
    print(batch, '\n')
    break

print('Data shapes and types in one batch:')
for batch in train_loader:
    for key in batch:
        print(f'{key[0]:15} {repr(key[1].shape):30} {key[1].dtype}')
    break

# %% [markdown]
# # Choose device

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda available')
else:
    device = torch.device('cpu')
    print('cuda not available')

# # for debugging purposes, force CPU
# device = torch.device('cpu')
# print('Warning! Forcing CPU device for debugging purposes.')

mlflow.log_param('device', device)

# %% [markdown]
# # Define model

# %%
# get artifact path (where mlflow saves artifacts)

art_path = mlflow.get_artifact_uri()
art_path = art_path[8:]  # remove 'file:///' from path
# turn %20 into spaces
art_path = urllib.parse.unquote(art_path, encoding=None, errors=None)

# if there is no drive letter, prepend another slash (e.g. for linux)
if os.path.splitdrive(art_path)[0] == '':
    art_path = '/' + art_path

print('art_path:', art_path)


# %%
# Save this script itself
file_path = os.path.abspath(__file__)

# try:
if True:
    print(file_path)
    mlflow.log_artifact(file_path)
    print('Logged script')
# except Exception as e:
#     print(f'Could not log script: {repr(e)}')
#     print('Continuing without logging script...')

# %%
if network_type == 'GNNtimeConv':
    import model_definitions.GNNtimeConv_def as model_def
elif network_type == 'EGNNtimeConv':
    import model_definitions.EGNNtimeConv_def as model_def

# # Save model definition
module_path = os.path.abspath(model_def.__file__)
mlflow.log_artifact(module_path)
print('Logged model definition')

# %% [markdown]
# # Test prior

# %%
# batch

# # %%
# import importlib
# importlib.reload(model_def)

prior_type = 'Prior'  # 'Prior' or 'Prior_wide'
mlflow.log_param('prior_type', prior_type)

if prior_type == 'Prior_wide':
    mu_phi, std_phi = -0.00030389729903857826, 0.014231439025614998
    mu_eps, std_eps = 2.0113846990910993e-05, 0.008196196348061435
    a, b = 4.042343191342216e-05, 0.0025632507
    prior = model_def.Prior_wide(std_phi=std_phi, mu_eps=mu_eps, std_eps=std_eps, a=a, b=b)

elif prior_type == 'Prior':
    lamb = 94.09494942436929
    mu_eps = 2.0113846990910993e-05
    std_eps = 0.008196196348061435
    prior = model_def.Prior(lamb=lamb, mu_eps=mu_eps, std_eps=std_eps)

print(prior)

for i, batch in enumerate(train_loader):
    batch = batch.to(device)
    e = batch.edge_index
    L_init_temp = batch.L_init[e[0] == e[1]-1]  # use only the beam element edges, not the virtual ones, not the reversed ones

    print(batch)
    print('L_init_temp.shape', L_init_temp.shape)

    t1 = time.time()
    x_0 = prior(batch.N, batch.d, batch.node_attr[..., :3], L_init_temp, batch.batch, batch.L_tot)
    print('time for prior:', time.time() - t1)
    print('x_0.shape', x_0.shape)

    break


# %%
ind = 3 # index of the graph in the batch
pos = x_0[batch.batch == ind].cpu().numpy()
L_tot = batch.L_tot[ind].cpu().numpy().item()  # y-coordinate of last node in first time step

# %%
# Plot the sampled beam trajectory, all time steps in one figure

# margin = 0.25
# factor = margin + 1.0

# fig, ax = plt.subplots(figsize=(5,3), dpi=200)

# # fig.subplots_adjust(top=0.8)

# for i in range(pos.shape[-1]):  # loop over time steps
#     for j in range(len(pos)-1): # plot segments
#         ax.plot(pos[j:j+2, 0, i], pos[j:j+2, 1, i],
#                 color=plt.cm.viridis(i / (pos.shape[-1] - 1)))
#     ax.scatter(pos[:, 0, i], pos[:, 1, i], s=3, color='black')

# ax.add_patch(plt.Rectangle((-L_tot * factor, -L_tot * factor), 2 * L_tot * factor, L_tot * factor, facecolor='gray', linewidth=0))

# ax.set_xlim(-L_tot * factor, L_tot * factor)
# ax.set_ylim(-L_tot * factor*0.1, L_tot * factor)

# ax.set_aspect('equal')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()
# plt.savefig('test.png', dpi=200)
# plt.close()

# %%
# plt.figure(figsize=(4,4), dpi=200)
# plt.plot(pos[:, 0].T)
# plt.xlabel('Time step')
# plt.ylabel('x-coordinates')

# # %%
# plt.figure(figsize=(4,4), dpi=200)
# plt.plot(pos[:, 1].T)
# plt.xlabel('Time step')
# plt.ylabel('y-coordinates')

# %% [markdown]
# # Define and create Flow

# %%
# import importlib
# importlib.reload(model_def)


# %%


# %%
class Flow(nn.Module):
    def __init__(self):
        super(Flow, self).__init__()
        if network_type == 'GNNtimeConv':
            # layers =  [(4, 3, 1, 2, 0)]
            # node_in, edge_in, message_size, node_out, edge_out:
            layers =  [(4, 3, 32, 32, 32),
                       (32, 32, 32, 32, 32),
                       (32, 32, 1, 2, 0)
                       ]

            # reuse_layers = (1,)
            reuse_layers = (1,1,1)
            self.model = model_def.GNN(layers=layers, reuse_layers=reuse_layers).to(device)
        elif network_type == 'EGNNtimeConv':
            # layers =  [(4, 3, 1, 0, 0)]
            layers =  [(4, 3, 32, 32, 32),
                       (32, 32, 32, 32, 32),
                       (32, 32, 1, 0, 0)
                       ]
            # reuse_layers = (1,)
            reuse_layers = (1,1,1)
            self.model = model_def.EGNN(layers=layers, reuse_layers=reuse_layers).to(device)

    def forward(self, t, x_t, batch, verbose=False) -> Tensor:
        # t: torch.tensor, shape [batch size,], current flow-matching time
        # x_t: torch.tensor, shape [batch size, 2, T], current position

        x_shift = self.model(batch=batch, current_pos=x_t, tau=t, verbose=verbose)

        return x_shift

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, batch) -> Tensor:
        # t_start: float, current time
        # t_end: float, end time
        # x_t: shape [batch size, 2, T], current position

        t_start = t_start.expand(x_t.shape[0]).view(-1, 1, 1)
        t_end = t_end.expand(x_t.shape[0]).view(-1, 1, 1)

        return (x_t + (t_end - t_start)
                * self(
                    t=(t_start + (t_end - t_start) / 2).view(-1),
                    x_t= x_t + self(x_t=x_t, t=t_start.view(-1), batch=batch) * (t_end - t_start) / 2,
                    batch=batch
                        )
                )

# %%
flow = Flow().to(device)
print(flow)

n_params = sum(p.numel() for p in flow.parameters())
print('Total nr of parameters:', n_params)
mlflow.log_param('n_params', n_params)

# %%
# Check if the model can run on a batch
for j, batch in enumerate(train_loader):
    print('Batch:', j)
    batch = batch.to(device)
    e = batch.edge_index
    L_init_temp = batch.L_init[e[0] == e[1]-1]  # use only the beam element edges, not the virtual ones, not the reversed ones

    x_0 = prior(batch.N, batch.d, batch.node_attr[..., :3], L_init_temp, batch.batch, batch.L_tot)

    x_1 = batch.pos[..., 0]  # final position (target)

    n_batches = torch.max(batch.batch) + 1

    t = torch.rand(n_batches).to(device)

    x_t = (1 - t[batch.batch].reshape(-1, 1, 1)) * x_0 + t[batch.batch].reshape(-1, 1, 1) * x_1

    print(flow(t=t, x_t=x_t, batch=batch))

    break

# %% [markdown]
# # Train

# %%

def MSELoss_allTargets(pred, target, batch, return_indices=False):
    """MSE loss applied on all possible targets, then taking the minimum per graph

    Parameters
    ----------
    pred : torch tensor, [n, f]
        prediction per node, n = total nr of nodes, f = nr of predicted features
    target : torch tensor, [n, f, a]
        all possible targets, n = total nr of nodes, f = nr of predicted features, a = nr of possible alternatives
    batch : torch tensor, [n, ]
        graph that each node belongs to
    return_indices : bool, optional
        whether to return the indices of the alternatives giving the minimum loss, by default False

    Returns
    -------
    torch tensor, [1,]
        MSE loss, using the closest alternative for each graph
    torch tensor, [G,], where G is the nr of graphs
        indices of the alternatives giving the minimum loss per graph, if return_indices is True.
    """
    # sum square error per node per alternative (resulting shape: [n, a])
    SE = torch.sum((pred.unsqueeze(-1) - target)**2, dim=1)
    # sum square error per graph per alternative (resulting shape: [G, a], with G the nr of graphs)
    SE = tg.nn.global_add_pool(SE, batch=batch)
    # take the minimum error per graph
    SE_min, inds = torch.min(SE, axis=-1)

    # take mean over entire batch
    MSE = torch.sum(SE_min)/(pred.numel())

    if return_indices:
        return MSE, inds
    else:
        return MSE

# %%
optimizer = torch.optim.Adam(flow.parameters(), 1e-3)
loss_fn = nn.MSELoss()

grad_clipping = 0.5
mlflow.log_param('grad_clipping', grad_clipping)

symmetric_matching = True
mlflow.log_param('symmetric_matching', symmetric_matching)

MSE_loss = []
MSE_val = []
n_epochs = 2000
for i in range(n_epochs):
    print('Epoch:', i)
    if i%100 == 0:
        print(i)
    loss_temp = []
    for j, batch in enumerate(train_loader):
        # print('Batch:', j)
        batch = batch.to(device)
        e = batch.edge_index
        L_init_temp = batch.L_init[e[0] == e[1]-1]  # use only the beam element edges, not the virtual ones, not the reversed ones

        x_0 = prior(batch.N, batch.d, batch.node_attr[..., :3], L_init_temp, batch.batch, batch.L_tot)

        if symmetric_matching:
            # for each graph, find the closest target positions to train on
            _, inds = MSELoss_allTargets(x_0.reshape(x_0.shape[0], -1), batch.pos.reshape(batch.pos.shape[0], -1, 2), batch.batch, return_indices=True)
            x_1 = torch.take_along_dim(batch.pos, inds[batch.batch].reshape(-1, 1, 1, 1), dim=-1).squeeze(-1)  # final position (target)
        else:
            x_1 = batch.pos[..., 0]

        n_graphs = torch.max(batch.batch) + 1

        t = torch.rand(n_graphs).to(device)

        x_t = (1 - t.reshape(-1, 1, 1)[batch.batch]) * x_0 + t.reshape(-1, 1, 1)[batch.batch] * x_1
        dx_t = x_1 - x_0

        optimizer.zero_grad()
        loss = loss_fn(flow(t=t, x_t=x_t, batch=batch), dx_t)

        if loss.isnan():
            print('Warning: loss is NaN')
            raise ValueError('Loss is NaN.')

        for name, param in flow.named_parameters():
            if param.isnan().any():
                print(f'Warning: parameter {name} has NaN values before backward and optimizer.step().')
                raise ValueError(f'Parameter {name} has NaN values.')
            if param.grad is not None and param.grad.isnan().any():
                print(f'Warning: gradient of parameter {name} has NaN values before backward and optimizer.step().')
                raise ValueError(f'Gradient of parameter {name} has NaN values.')

        # torch.autograd.set_detect_anomaly(True)
        loss.backward()

        # to do: fix: parameters get NaNs gradients somewhere here, but no NaNs in the loss or the intermediate values anywhere
        raise_grad_error = False
        for name, param in flow.named_parameters():
            if param.isnan().any():
                # print(f'! {name}     : has NaNs')
                # print(f'Warning: parameter {name} has NaN values after backward and before optimizer.step().')
                raise_grad_error = True
            if param.grad is not None and param.grad.isnan().any():
                # print(f'! {name}.grad: has NaNs')
                # print(f'Warning: gradient of parameter {name} has NaN values after backward and before optimizer.step().')
                raise_grad_error = True
        if raise_grad_error:
            raise ValueError('NaN values detected in parameters or gradients after backward and before optimizer.step().')

        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clipping)

        optimizer.step()

        for name, param in flow.named_parameters():
            if param.isnan().any():
                print(f'Warning: parameter {name} has NaN values.')
                raise ValueError(f'Parameter {name} has NaN values after optimizer.step().')
            if param.grad is not None and param.grad.isnan().any():
                print(f'Warning: gradient of parameter {name} has NaN values.')
                raise ValueError(f'Gradient of parameter {name} has NaN values after optimizer.step().')

        loss_temp.append(loss.item())

    MSE_loss.append(np.mean(loss_temp))
    mlflow.log_metric('MSE_train', np.mean(loss_temp), step=i)
    print('Epoch loss:', np.mean(loss_temp))

    if i == 1800:
        optimizer = torch.optim.Adam(flow.parameters(), 1e-4)


    loss_temp = []
    for j, batch in enumerate(test_loader):
        with torch.no_grad():
            # print('Batch:', j)
            batch = batch.to(device)
            e = batch.edge_index
            L_init_temp = batch.L_init[e[0] == e[1]-1]  # use only the beam element edges, not the virtual ones, not the reversed ones

            x_0 = prior(batch.N, batch.d, batch.node_attr[..., :3], L_init_temp, batch.batch, batch.L_tot)

            # for each graph, find the closest target positions to test on
            _, inds = MSELoss_allTargets(x_0.reshape(x_0.shape[0], -1), batch.pos.reshape(batch.pos.shape[0], -1, 2), batch.batch, return_indices=True)
            x_1 = torch.take_along_dim(batch.pos, inds[batch.batch].reshape(-1, 1, 1, 1), dim=-1).squeeze(-1)  # final position (target)

            n_graphs = torch.max(batch.batch) + 1

            t = torch.rand(n_graphs).to(device)

            x_t = (1 - t.reshape(-1, 1, 1)[batch.batch]) * x_0 + t.reshape(-1, 1, 1)[batch.batch] * x_1
            dx_t = x_1 - x_0

            loss = loss_fn(flow(t=t, x_t=x_t, batch=batch), dx_t)
            loss_temp.append(loss.item())

    MSE_val.append(np.mean(loss_temp))
    mlflow.log_metric('MSE_val', np.mean(loss_temp), step=i)
    print('Validation loss:', np.mean(loss_temp))

    if i % 20 == 0 or i==n_epochs-1:
        # Do full inference on the validation data
        time_steps = torch.linspace(0, 1, 16).to(device)  # 16 time steps from 0 to 1

        mse_full_all = []
        for j, batch in enumerate(test_loader):
            with torch.no_grad():
                # print('Batch:', j)
                batch = batch.to(device)

                e = batch.edge_index
                L_init_temp = batch.L_init[e[0] == e[1]-1]  # use only the beam element edges, not the virtual ones, not the reversed ones

                x = prior(batch.N, batch.d, batch.node_attr[..., :3], L_init_temp, batch.batch, batch.L_tot)

                real = batch.pos  # final position (target)

                for k in range(15):
                    # print(f'Step {i}/{n_steps}')

                    x = flow.step(x_t=x, t_start=time_steps[k], t_end=time_steps[k + 1], batch=batch)

                mse_full_all_temp = MSELoss_allTargets(x.reshape(x.shape[0], -1), real.reshape(real.shape[0], -1, 2), batch.batch, return_indices=False)
                mse_full_all.append(mse_full_all_temp.item())

                if mse_full_all_temp.item() > 10 or np.isnan(mse_full_all_temp.item()):
                    print(f'Warning: MSE full inference for batch {j} is high or NaN: {mse_full_all_temp.item()}')
                    # raise ValueError('Full MSE too high or NaN.')

        mse_full_all = np.mean(mse_full_all)
        print('Full inference MSE:', mse_full_all)
        mlflow.log_metric('MSE_val_full', mse_full_all, step=i)


# %%
# get mlflow run ID
run_id = mlflow.active_run().info.run_id
run_id

# %%
plt.figure(figsize=(7, 4), dpi=150)
plt.subplots_adjust(left=0.15, bottom=0.2)
plt.plot(MSE_loss, label='Training loss')
plt.plot(MSE_val, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.yscale('log')
plt.legend()

mlflow.log_figure(plt.gcf(), 'loss.png')

# plt.show()
plt.close()

# %% [markdown]
# # Save model

# %%
# %% Save things
# save parameters of model
try:
    path = os.path.join(art_path, 'model_weights.pt')
    torch.save(flow.state_dict(), path)
    print('Saved model parameters to', path)
except AttributeError as e:
    print('Error saving model parameters:', repr(e))

# # Save entire model
try:
    path = os.path.join(art_path, 'model.pt')
    torch.save(flow, path)
    print('Saved model to', path)
except AttributeError as e:
    print('Error saving model:', repr(e))
except pickle.PicklingError as e:
    print('Error saving model:', repr(e))

# %% [markdown]
# # Sampling

# %%
test_loader3 = tg.loader.DataLoader(data['data_te'], batch_size=64)

# %%
# clear CUDA cache
torch.cuda.empty_cache()

# %%
import matplotlib

# %%
n_steps = 8

fig, axes = plt.subplots(1, n_steps//2 + 1, figsize=(15, 4), sharex=True, sharey=True)

time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)

for ax in axes:
    # ax.set_aspect('equal')
    ax.set_xlabel('Real position')
    # ax.axline((0,0), (1,1), c='tab:red')
    # ax.axline((0,0), (-1,1), c='tab:red')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-20, 20)
axes[0].set_ylabel('Predicted position')

with torch.no_grad():
    for j, batch in enumerate(test_loader3):
        print('Batch:', j)
        batch = batch.to(device)

        e = batch.edge_index
        L_init_temp = batch.L_init[e[0] == e[1]-1]  # use only the beam element edges, not the virtual ones, not the reversed ones

        x = prior(batch.N, batch.d, batch.node_attr[..., :3], L_init_temp, batch.batch, batch.L_tot)

        real = batch.pos[..., 0].cpu().detach().numpy()  # final position (target)

        pred = x.cpu().detach().numpy()

        axes[0].scatter(real, pred, s=1, c='tab:blue', alpha=0.05)
        axes[0].set_title(r'$\tau$' + f' = {time_steps[0]:.2f}')

        for i in range(n_steps):
            # print(f'Step {i}/{n_steps}')

            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], batch=batch)

            if i % 2 == 1:
                pred = x.cpu().detach().numpy()

                # axes[i//2 + 1].plot(np.arange(20))
                axes[i//2 + 1].scatter(real, pred, s=1, c='tab:blue', alpha=0.05)
                axes[i//2 + 1].set_title(r'$\tau$' + f' = {time_steps[i + 1]:.2f}')

    # plt.tight_layout()

    mlflow.log_figure(plt.gcf(), 'real_vs_pred_pos.png')
    plt.close()

# %% [markdown]
# # One test graph

# %%
# release cuda memory
torch.cuda.empty_cache()


# %%
test_loader2_temp  = tg.loader.DataLoader(data['data_te'], batch_size=2)

# %%
batch = next(iter(test_loader2_temp))
batch = batch.to(device)

e = batch.edge_index
L_init_temp = batch.L_init[e[0] == e[1]-1]  # use only the beam element edges, not the virtual ones, not the reversed ones

# Sample from prior to create initial positions at tau=0
x_prior = prior(batch.N, batch.d, batch.node_attr[..., :3], L_init_temp, batch.batch, batch.L_tot)

L_tot = batch.L_tot[0].cpu().numpy().item()


# %%
# plt.figure(figsize=(4,4), dpi=200)
# plt.plot(x_prior[:, 0][batch.batch==0].cpu().detach().numpy().T)
# plt.xlabel('Time step')
# plt.ylabel('x-coordinates')

# %%
# plt.figure(figsize=(4,4), dpi=200)
# plt.plot(x_prior[:, 1][batch.batch==0].cpu().detach().numpy().T)
# plt.xlabel('Time step')
# plt.ylabel('y-coordinates')

# %%
# fig2, ax2 = plt.subplots(figsize=(4,4), dpi=200)
# ax2.plot(x_prior[:, 1][batch.batch==0].cpu().detach().numpy().T)
# ax2.set_xlabel('Time step')
# ax2.set_ylabel('y-coordinates')
# mlflow.log_figure(fig2, f'sampled_beam_prior2.png')

# %%
with torch.no_grad():
    for n_steps in [4, 16, 64, 256]:
        print(f'\n\nSampling with {n_steps} flow-matching steps')

        margin = 0.25
        factor = margin + 1.0

        time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
        print('len(time_steps)', len(time_steps))
        inds_to_plot = np.linspace(0, n_steps, 5).astype(int)
        print('inds_to_plot', inds_to_plot)

        fig, axes = plt.subplots(1, len(inds_to_plot)+1, figsize=(14, 2.5), sharex=True, sharey=True, dpi=100)

        fig.suptitle(f'Using {n_steps} flow-matching steps')

        for ax in axes:
            ax.add_patch(plt.Rectangle((-L_tot * factor, -L_tot * factor), 2 * L_tot * factor, L_tot * factor, facecolor='gray', linewidth=0))

            ax.set_xlim(-L_tot * factor, L_tot * factor)
            ax.set_ylim(-L_tot * factor*0.1, L_tot * factor)

            ax.set_aspect('equal')

            ax.set_xlabel('x')
        axes[0].set_ylabel('y')

        # Plot initial positions
        pred = x_prior[batch.batch==0].cpu().detach()
        for t in range(pred.shape[-1]):  # loop over time steps
            for j in range(len(pred)-1): # plot segments
                axes[0].plot(pred[j:j+2, 0, t], pred[j:j+2, 1, t],
                        color=plt.cm.viridis(t / (pred.shape[-1] - 1)))
            axes[0].scatter(pred[:, 0, t], pred[:, 1, t], s=3, color='black')
        axes[0].set_title(r'$\tau$' + f' = {time_steps[0]:.2f}')

        # Plot real positions
        real = batch.pos[..., 0][batch.batch==0].cpu().detach().numpy()  # final position (target)
        for t in range(real.shape[-1]):  # loop over time steps
            for j in range(len(real)-1): # plot segments
                axes[-1].plot(real[j:j+2, 0, t], real[j:j+2, 1, t],
                        color=plt.cm.viridis(t / (real.shape[-1] - 1)))
            axes[-1].scatter(real[:, 0, t], real[:, 1, t], s=3, color='black')
        axes[-1].set_title(r'Ground truth')

        to_plot = 1
        x = x_prior.clone() # copy to avoid modifying the original tensor
        for i in range(n_steps):  # loop over flow-matching steps
            print(f'Step {i}, from t={time_steps[i]:.2f} to t={time_steps[i + 1]:.2f}')
            # Update positions through flow-matching steps
            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], batch=batch)

            # Plot some of the steps
            if i+1 == inds_to_plot[to_plot]:
                print(f'Plot {to_plot}, Plotting step {i}, at time {time_steps[i + 1]}')
                pred = x[batch.batch==0].cpu().detach()

                for t in range(pred.shape[-1]):  # loop over time steps
                    for j in range(len(pred)-1): # loop over segments
                        axes[to_plot].plot(pred[j:j+2, 0, t], pred[j:j+2, 1, t],
                                color=plt.cm.viridis(t / (pred.shape[-1] - 1)))
                    axes[to_plot].scatter(pred[:, 0, t], pred[:, 1, t], s=3, color='black')
                axes[to_plot].set_title(r'$\tau$' + f' = {time_steps[i + 1]:.2f}')

                # fig2, ax2 = plt.subplots(figsize=(4,4), dpi=200)
                # ax2.plot(x[:, 1][batch.batch==0].cpu().detach().numpy().T)
                # ax2.set_xlabel('Time step')
                # ax2.set_ylabel('y-coordinates')
                # mlflow.log_figure(fig2, f'sampled_beam_{n_steps}_FMsteps_tau{time_steps[i + 1]:.2f}.png')

                to_plot += 1

        plt.tight_layout()

        mlflow.log_figure(fig, f'sampled_beam_{n_steps}_FMsteps_prior.png')

        # plt.show()
        plt.close()


# %%
T = batch.d.shape[-1]  # nr of time steps
print('T:', T)

t_plot = np.linspace(0, T-1, 20, dtype=int)  # 20 time steps to plot
print('t_plot:', t_plot)

# %%
# same plots, fewer time steps

with torch.no_grad():
    for n_steps in [4, 16, 64, 256]:
        print(f'\n\nSampling with {n_steps} flow-matching steps')

        margin = 0.25
        factor = margin + 1.0

        time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
        print('len(time_steps)', len(time_steps))
        inds_to_plot = np.linspace(0, n_steps, 5).astype(int)
        print('inds_to_plot', inds_to_plot)

        fig, axes = plt.subplots(1, len(inds_to_plot)+1, figsize=(14, 2.5), sharex=True, sharey=True, dpi=100)

        fig.suptitle(f'Using {n_steps} flow-matching steps')

        for ax in axes:
            ax.add_patch(plt.Rectangle((-L_tot * factor, -L_tot * factor), 2 * L_tot * factor, L_tot * factor, facecolor='gray', linewidth=0))

            ax.set_xlim(-L_tot * factor, L_tot * factor)
            ax.set_ylim(-L_tot * factor*0.1, L_tot * factor)

            ax.set_aspect('equal')

            ax.set_xlabel('x')
        axes[0].set_ylabel('y')

        # Plot initial positions
        pred = x_prior[batch.batch==0].cpu().detach().numpy()
        pred = pred[:, :, t_plot]  # only plot the time steps we want
        for t in range(pred.shape[-1]):  # loop over time steps
            for j in range(len(pred)-1): # plot segments
                axes[0].plot(pred[j:j+2, 0, t], pred[j:j+2, 1, t],
                        color=plt.cm.viridis(t / (pred.shape[-1] - 1)))
            axes[0].scatter(pred[:, 0, t], pred[:, 1, t], s=3, color='black')
        axes[0].set_title(r'$\tau$' + f' = {time_steps[0]:.2f}')

        # Plot real positions
        real = batch.pos[..., 0][batch.batch==0].cpu().detach().numpy()  # final position (target)
        real = real[:, :, t_plot]  # only plot the time steps we want
        for t in range(real.shape[-1]):  # loop over time steps
            for j in range(len(real)-1): # plot segments
                axes[-1].plot(real[j:j+2, 0, t], real[j:j+2, 1, t],
                        color=plt.cm.viridis(t / (real.shape[-1] - 1)))
            axes[-1].scatter(real[:, 0, t], real[:, 1, t], s=3, color='black')
        axes[-1].set_title(r'Ground truth')

        to_plot = 1
        x = x_prior.clone() # copy to avoid modifying the original tensor
        for i in range(n_steps):  # loop over flow-matching steps
            print(f'Step {i}, from t={time_steps[i]:.2f} to t={time_steps[i + 1]:.2f}')
            # Update positions through flow-matching steps
            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], batch=batch)

            # Plot some of the steps
            if i+1 == inds_to_plot[to_plot]:
                print(f'Plot {to_plot}, Plotting step {i}, at time {time_steps[i + 1]}')
                pred = x[batch.batch==0].cpu().detach().numpy()
                pred = pred[:, :, t_plot]

                for t in range(pred.shape[-1]):  # loop over time steps
                    for j in range(len(pred)-1): # loop over segments
                        axes[to_plot].plot(pred[j:j+2, 0, t], pred[j:j+2, 1, t],
                                color=plt.cm.viridis(t / (pred.shape[-1] - 1)))
                    axes[to_plot].scatter(pred[:, 0, t], pred[:, 1, t], s=3, color='black')
                axes[to_plot].set_title(r'$\tau$' + f' = {time_steps[i + 1]:.2f}')

                to_plot += 1

        plt.tight_layout()

        mlflow.log_figure(fig, f'sampled_beam_{n_steps}_FMsteps_fewertimesteps.png')

        # plt.show()
        plt.close()


# %%
# 256 steps, fewer time steps, only plot prior, last step and real positions
with torch.no_grad():
    n_steps = 256

    print(f'\n\nSampling with {n_steps} flow-matching steps')

    margin = 0.25
    factor = margin + 1.0

    time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
    print('len(time_steps)', len(time_steps))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True, dpi=100)

    fig.suptitle(f'Using {n_steps} flow-matching steps')

    for ax in axes:
        ax.add_patch(plt.Rectangle((-L_tot * factor, -L_tot * factor), 2 * L_tot * factor, L_tot * factor, facecolor='gray', linewidth=0))

        ax.set_xlim(-L_tot * factor, L_tot * factor)
        ax.set_ylim(-L_tot * factor*0.1, L_tot * factor)

        ax.set_aspect('equal')

        ax.set_xlabel('x')
    axes[0].set_ylabel('y')

    # Plot initial positions
    pred = x_prior[batch.batch==0].cpu().detach().numpy()
    pred = pred[:, :, t_plot]  # only plot the time steps we want
    for t in range(pred.shape[-1]):  # loop over time steps
        for j in range(len(pred)-1): # plot segments
            axes[0].plot(pred[j:j+2, 0, t], pred[j:j+2, 1, t],
                    color=plt.cm.viridis(t / (pred.shape[-1] - 1)))
        axes[0].scatter(pred[:, 0, t], pred[:, 1, t], s=3, color='black')
    axes[0].set_title(r'$\tau$' + f' = {time_steps[0]:.2f}')

    # Plot real positions
    real = batch.pos[..., 0][batch.batch==0].cpu().detach().numpy()  # final position (target)
    real = real[:, :, t_plot]  # only plot the time steps we want
    for t in range(real.shape[-1]):  # loop over time steps
        for j in range(len(real)-1): # plot segments
            axes[-1].plot(real[j:j+2, 0, t], real[j:j+2, 1, t],
                    color=plt.cm.viridis(t / (real.shape[-1] - 1)))
        axes[-1].scatter(real[:, 0, t], real[:, 1, t], s=3, color='black')
    axes[-1].set_title(r'Ground truth')

    x = x_prior.clone() # copy to avoid modifying the original tensor
    for i in range(n_steps):  # loop over flow-matching steps
        # print(f'Step {i}, from t={time_steps[i]:.2f} to t={time_steps[i + 1]:.2f}')
        # Update positions through flow-matching steps
        x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], batch=batch)

    # Plot final sample
    pred = x[batch.batch==0].cpu().detach().numpy()
    pred = pred[:, :, t_plot]

    for t in range(pred.shape[-1]):  # loop over time steps
        for j in range(len(pred)-1): # loop over segments
            axes[1].plot(pred[j:j+2, 0, t], pred[j:j+2, 1, t],
                    color=plt.cm.viridis(t / (pred.shape[-1] - 1)))
        axes[1].scatter(pred[:, 0, t], pred[:, 1, t], s=3, color='black')
    axes[1].set_title(r'$\tau$' + f' = {time_steps[i + 1]:.2f}')

    plt.tight_layout()

    mlflow.log_figure(fig, f'sampled_beam_{n_steps}_FMsteps_fewertimesteps_bigger.png')

    # plt.show()
    plt.close()



# %% [markdown]
# # Multiple prior samples

# %%
batch = next(iter(test_loader2_temp))
batch = batch.to(device)

e = batch.edge_index
L_init_temp = batch.L_init[e[0] == e[1]-1]  # use only the beam element edges, not the virtual ones, not the reversed ones


L_tot = batch.L_tot[0].cpu().numpy().item()


# %%
# 64 steps, fewer time steps, only plot prior, last step and real positions
# multiple prior samples

n_steps = 64

with torch.no_grad():
    for k in range(8):

        # Sample from prior to create initial positions at tau=0
        x_prior = prior(batch.N, batch.d, batch.node_attr[..., :3], L_init_temp, batch.batch, batch.L_tot)

        print(f'\n\nSampling with {n_steps} flow-matching steps')

        margin = 0.25
        factor = margin + 1.0

        time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
        print('len(time_steps)', len(time_steps))

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True, dpi=100)

        fig.suptitle(f'Using {n_steps} flow-matching steps')

        for ax in axes:
            ax.add_patch(plt.Rectangle((-L_tot * factor, -L_tot * factor), 2 * L_tot * factor, L_tot * factor, facecolor='gray', linewidth=0))

            ax.set_xlim(-L_tot * factor, L_tot * factor)
            ax.set_ylim(-L_tot * factor*0.1, L_tot * factor)

            ax.set_aspect('equal')

            ax.set_xlabel('x')
        axes[0].set_ylabel('y')

        # Plot initial positions
        pred = x_prior[batch.batch==0].cpu().detach().numpy()
        pred = pred[:, :, t_plot]  # only plot the time steps we want
        for t in range(pred.shape[-1]):  # loop over time steps
            for j in range(len(pred)-1): # plot segments
                axes[0].plot(pred[j:j+2, 0, t], pred[j:j+2, 1, t],
                        color=plt.cm.viridis(t / (pred.shape[-1] - 1)))
            axes[0].scatter(pred[:, 0, t], pred[:, 1, t], s=3, color='black')
        axes[0].set_title(r'$\tau$' + f' = {time_steps[0]:.2f}')

        # Plot real positions
        real = batch.pos[..., 0][batch.batch==0].cpu().detach().numpy()  # final position (target)
        real2 = real[:, :, t_plot]  # only plot the time steps we want
        for t in range(real2.shape[-1]):  # loop over time steps
            for j in range(len(real2)-1): # plot segments
                axes[-1].plot(real2[j:j+2, 0, t], real2[j:j+2, 1, t],
                        color=plt.cm.viridis(t / (real2.shape[-1] - 1)))
            axes[-1].scatter(real2[:, 0, t], real2[:, 1, t], s=3, color='black')
        axes[-1].set_title(r'Ground truth')

        x = x_prior.clone() # copy to avoid modifying the original tensor
        for i in range(n_steps):  # loop over flow-matching steps
            # print(f'Step {i}, from t={time_steps[i]:.2f} to t={time_steps[i + 1]:.2f}')
            # Update positions through flow-matching steps
            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], batch=batch)

        # Plot final sample
        pred = x[batch.batch==0].cpu().detach().numpy()
        pred2 = pred[:, :, t_plot]

        for t in range(pred2.shape[-1]):  # loop over time steps
            for j in range(len(pred2)-1): # loop over segments
                axes[1].plot(pred2[j:j+2, 0, t], pred2[j:j+2, 1, t],
                        color=plt.cm.viridis(t / (pred2.shape[-1] - 1)))
            axes[1].scatter(pred2[:, 0, t], pred2[:, 1, t], s=3, color='black')
        axes[1].set_title(r'$\tau$' + f' = {time_steps[i + 1]:.2f}')
        plt.tight_layout()
        mlflow.log_figure(fig, f'sampled_beam_{n_steps}_FMsteps_fewertimesteps_bigger_prior{k}.png')
        # plt.show()
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(4,4), dpi=200)
        ax2.plot(real[:, 0].T, label='Real', color='tab:blue')
        ax2.plot(pred[:, 0].T, label='Predicted', color='tab:orange')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('x-coordinates')
        ax2.legend()
        plt.tight_layout()
        mlflow.log_figure(fig2, f'x_{n_steps}_FMsteps_fewertimesteps_bigger_prior{k}.png')
        # plt.show()
        plt.close()

        fig3, ax3 = plt.subplots(figsize=(4,4), dpi=200)
        ax3.plot(real[:, 1].T, label='Real', color='tab:blue')
        ax3.plot(pred[:, 1].T, label='Predicted', color='tab:orange')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('y-coordinates')
        ax3.legend()
        plt.tight_layout()
        mlflow.log_figure(fig3, f'y_{n_steps}_FMsteps_fewertimesteps_bigger_prior{k}.png')
        # plt.show()
        plt.close()


# %% [markdown]
# # x,y coordinates over tau

# %%
# 64 steps, x and y coordinates over time for all tau

n_steps = 64

with torch.no_grad():
    for k in range(8):

        fig2, ax2 = plt.subplots(figsize=(4,4), dpi=200)
        fig3, ax3 = plt.subplots(figsize=(4,4), dpi=200)

        # Sample from prior to create initial positions at tau=0
        x_prior = prior(batch.N, batch.d, batch.node_attr[..., :3], L_init_temp, batch.batch, batch.L_tot)

        print(f'\n\nSampling with {n_steps} flow-matching steps')

        margin = 0.25
        factor = margin + 1.0

        time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
        print('len(time_steps)', len(time_steps))

        # Plot initial positions
        pred = x_prior[batch.batch==0].cpu().detach().numpy()
        ax2.plot(pred[:, 0].T, label='Predicted', color=plt.cm.viridis(0))
        ax3.plot(pred[:, 1].T, label='Predicted', color=plt.cm.viridis(0))

        # Plot real positions
        real = batch.pos[..., 0][batch.batch==0].cpu().detach().numpy()  #
        ax2.plot(real[:, 0].T, label='Real', color='tab:red')
        ax3.plot(real[:, 1].T, label='Real', color='tab:red')

        x = x_prior.clone() # copy to avoid modifying the original tensor
        for i in range(n_steps):  # loop over flow-matching steps
            # print(f'Step {i}, from t={time_steps[i]:.2f} to t={time_steps[i + 1]:.2f}')
            # Update positions through flow-matching steps
            x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1], batch=batch)

            # Plot final sample
            pred = x[batch.batch==0].cpu().detach().numpy()

            ax2.plot(pred[:, 0].T, label='Predicted', color=plt.cm.viridis((i + 1) / n_steps))
            ax3.plot(pred[:, 1].T, label='Predicted', color=plt.cm.viridis((i + 1) / n_steps))

        ax2.set_xlabel('Time step')
        ax2.set_ylabel('x-coordinates')
        # ax2.legend()
        plt.tight_layout()
        mlflow.log_figure(fig2, f'x_prior{k}.png')

        ax3.set_xlabel('Time step')
        ax3.set_ylabel('y-coordinates')
        # ax3.legend()
        plt.tight_layout()
        mlflow.log_figure(fig3, f'y_prior{k}.png')
        # plt.show()
        plt.close()


# end mlflow run
mlflow.end_run()