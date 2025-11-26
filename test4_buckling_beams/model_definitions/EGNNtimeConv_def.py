"""EGNNtimeConv_def.py but with EGNN over time (which I deactivated for now)."""
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg
import numpy as np
# import model_definitions.unet as unet

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, num_layers=4, kernel_size=6, resize_factor=3):
        super(UNet1D, self).__init__()
        self.depth = num_layers

        if isinstance(resize_factor, int):
            self.resize_factor = [resize_factor]*num_layers
        else:
            assert len(resize_factor) == num_layers, "resize_factor must be an int or a list of length num_layers"
            self.resize_factor = resize_factor

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for i in range(num_layers):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(prev_channels, hidden_channels, kernel_size=kernel_size, padding='same'),
                    nn.ReLU()
                )
            )
            self.pools.append(nn.MaxPool1d(kernel_size=self.resize_factor[i], stride=self.resize_factor[i]))
            prev_channels = hidden_channels

        self.middle_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='same')

        # Decoder path
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            # After concatenation, channels = hidden_channels * 2 except at final
            in_ch = hidden_channels * 2  # if i < num_layers - 1 else hidden_channels
            self.decoders.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, hidden_channels, kernel_size=kernel_size, padding='same'),
                    nn.ReLU()
                )
            )

        self.extra_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='same')

        # Final mapping
        self.final_convs = nn.Sequential(
            nn.Conv1d(hidden_channels + in_channels, hidden_channels, kernel_size=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1, padding='same')
        )

    def forward(self, tau, x, verbose=False):
        if torch.isnan(x).any():
            raise ValueError('NaN values encountered in input x of UNet1D forward pass.')
        if torch.isnan(tau).any():
            raise ValueError('NaN values encountered in input tau of UNet1D forward pass.')

        # Encoder
        x_in = x.clone()
        skips = []
        if verbose:
            print(f'{"Input shape         :":<30} {x.shape}')
        for i, [encoder, pool] in enumerate(zip(self.encoders, self.pools)):
            x = encoder(x)
            skips.append(x)
            if verbose:
                print(f'{"Skip shape          :":<30} {x.shape}')

            x = pool(x)

            if verbose:
                print(f'{"Encoder output shape:":<30} {x.shape}')

            if torch.isnan(x).any():
                raise ValueError(f'NaN values encountered in UNet1D encoder {i} forward pass.')


        x = self.middle_conv(x)
        x = F.relu(x)

        if torch.isnan(x).any():
            raise ValueError('NaN values encountered in UNet1D middle conv forward pass.')

        if verbose:
            print(f'{"Middle conv shape   :":<30} {x.shape}')

        # Bottleneck output
        # Decoder
        for i in range(0, len(self.decoders)):
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-1], mode='linear')
                              #scale_factor=self.resize_factor[-(i+1)], mode='linear')
            if verbose:
                print(f'{"Decoder interpolated shape :":<30} {x.shape}')
            if torch.isnan(x).any():
                raise ValueError('NaN values encountered in UNet1D decoder after interpolation step.')
            # # Skip connection: concatenate with corresponding encoder output
            # skip = skips.pop()
            # # If sizes mismatch, pad
            # if x.shape[-1] != skip.shape[-1]:
            #     if verbose:
            #         print(f'Size mismatch, padding needed (x.shape[-1]={x.shape[-1]}, skip.shape[-1]={skip.shape[-1]})')
            #     diff = skip.shape[-1] - x.shape[-1]
            #     pad_left = diff // 2
            #     pad_right = diff - pad_left
            #     x = F.pad(x, (pad_left, pad_right))

            x = torch.cat([x, skip], dim=1)  # if i < len(self.decoders) - 1 else x
            x = self.decoders[i](x)
            if verbose:
                print(f'{"Decoder output shape :":<30} {x.shape}')
            if torch.isnan(x).any():
                raise ValueError(f'NaN values encountered in UNet1D decoder after skip connection step {i}')

        x = self.extra_conv(x)
        x = torch.cat((x, x_in), dim=1)  # concatenate with input for final mapping

        if torch.isnan(x).any():
            raise ValueError('NaN values encountered in UNet1D before final mapping.')

        x = self.final_convs(x)  # final mapping to output channels
        return x

def cos_sin_phi_from_pos(pos, node_type, batch=None):
    """Compute cos(phi) and sin(phi) from the positions of the nodes.
    Parameters
    ----------
    pos : torch.tensor, shape [n, 2, T]
        positions of the nodes, where n is the number of nodes and T is the number of time steps.
    node_type : torch.tensor, shape [n, 3]
        node type for each of the n nodes in a one-hot encoding, where:
        - node_type[:, 0] == 1: middle nodes in the beam (regular nodes)
        - node_type[:, 1] == 1: first node in the beam (start node)
        - node_type[:, 2] == 1: last node in the beam (end node)
    batch : torch.tensor, shape [n, 1], optional
        batch indices for each node, shape [n, 1]. This is used to select the correct beam for each node.
        If None, it is assumed that all nodes belong to the same beam. Default is None.

    Returns
    -------
    cos_phi : torch.tensor, shape [n, 1, T]
        cosine of the angle phi for each node and time step.
    sin_phi : torch.tensor, shape [n, 1, T]
        sine of the angle phi for each node and time step.
    """
    n, _, T = pos.shape  # n nodes, T time steps
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=pos.device)  # assume all nodes belong to the same beam

    # sort-of-edge quantities (entries where node_type == end node are meaningless)
    r = torch.empty((n, 2, T), device=pos.device, dtype=pos.dtype)  # edge vectors, shape [n, 2, T]
    r[:-1] = pos[1:] - pos[:-1]
    # Use sqrt(r^2 + eps) instead of norm for numerical stability in gradients
    eps = 1e-8
    d = torch.sqrt(torch.sum(r**2, dim=1, keepdim=True) + eps)  # shape [n, 1, T], distance for each segment
    cos_q = r[:, [1]]/d
    sin_q = r[:, [0]]/d

    cos_phi = torch.empty((n, 1, T), device=pos.device, dtype=pos.dtype)  # shape [n, 1, T]
    sin_phi = torch.empty((n, 1, T), device=pos.device, dtype=pos.dtype)  # shape [n, 1, T]

    # For node 0: cos_phi = cos_q, sin_phi = sin_q
    cos_phi[node_type[:, 1] == 1, :] = cos_q[node_type[:, 1] == 1]  # shape of selection: [G, 1, T]
    sin_phi[node_type[:, 1] == 1, :] = sin_q[node_type[:, 1] == 1]  # shape of selection: [G, 1, T]

    # Node 1 - N-1: cos_phi = cos(q_i - q_{i-1})
    cos_diff_q = torch.empty((n, 1, T), device=pos.device, dtype=pos.dtype)
    sin_diff_q = torch.empty((n, 1, T), device=pos.device, dtype=pos.dtype)
    cos_diff_q[1:] = cos_q[1:] * cos_q[:-1] + sin_q[1:] * sin_q[:-1]
    sin_diff_q[1:] = sin_q[1:] * cos_q[:-1] - cos_q[1:] * sin_q[:-1]
    cos_phi[[node_type[:, 0] == 1]] = cos_diff_q[[node_type[:, 0] == 1]]
    sin_phi[[node_type[:, 0] == 1]] = sin_diff_q[[node_type[:, 0] == 1]]

    # For last node: cos_phi = 1.0, sin_phi = 0.0
    cos_phi[node_type[:, 2] == 1, :] = 1.0
    sin_phi[node_type[:, 2] == 1, :] = 0.0

    return cos_phi, sin_phi

def limited_growth(x):
    return torch.sign(x)*torch.log(1 + torch.abs(x))

class EdgeNodeUpdate(tg.nn.MessagePassing):
    """message passing layer that updates: node embedding, edge embedding and returns the messages.
    The message each edge sends is based on the distance between and the embedding of the nodes it connects and the edge embedding.
    The node embedding is updated based on the previous node embedding and the aggregated message received from neighboring nodes.
    The edge embedding is updated based on the message it sends.

    parameters
    ----------
    messagepassing : [type]
        [description]
    """

    def __init__(self, node_in, edge_in, message_size, node_out, edge_out, hidden_channels=16):

        """initialize layer

        parameters
        ----------
        node_in : int
            previous node embedding size
        edge_in: int
            previous edge embedding size
        message_size : int
            size of the message
        node_out : int
            node embedding size after updating
        edge_out: int
            edge embedding size after updating
        """

        super().__init__(aggr='mean', node_dim=0)
        self.unet_message = UNet1D(7 + 2*node_in + edge_in, message_size, hidden_channels=hidden_channels)
        # self.unet_message = unet.UNetModel(in_channels=5 + 2*node_in + edge_in,
        #                                    model_channels=hidden_channels,
        #                                    out_channels=1,
        #                                    num_res_blocks=3,
        #                                    attention_resolutions=[2,],
        #                                    dims=1)
        if node_out == 0:
            self.unet_update = None
        else:
            self.unet_update = UNet1D(1 + node_in + message_size, node_out, hidden_channels=hidden_channels)
            # self.unet_update = unet.UNetModel(
            #     in_channels=1 + node_in + message_size,
            #     model_channels=hidden_channels,
            #     out_channels=node_out,
            #     num_res_blocks=3,
            #     attention_resolutions=[2,],
            #     dims=1)
        if edge_out == 0:
            self.unet_edge = None
        else:
            # self.mlp_edge = torch.nn.Sequential(
            #                 torch.nn.Linear(message_size, edge_out),
            #                 )
            self.unet_edge = UNet1D(message_size, edge_out, hidden_channels=hidden_channels)
            # self.unet_edge = unet.UNetModel(
            #     in_channels=message_size,
            #     model_channels=hidden_channels,
            #     out_channels=edge_out,
            #     num_res_blocks=3,
            #     attention_resolutions=[2,],
            #     dims=1)

        # self.dt_unet = UNet1D(5 + node_in, 2, hidden_channels=hidden_channels)

        self._message_forward_hooks['hook1'] = self.hook_temp
        self.relative_distance_calculator = RelativeDistance()

    def hook_temp(self, net, msg_kwargs, out):
        net.messages = out

    def forward(self, x, edge_index, edge_attr, r, L_init, node_type, tau, pos, batch, verbose=False):
        """n nodes, G graphs, E edges, T time steps, dim dimensions (always 2 for now).

        parameters
        ----------
        x : torch.tensor, shape [n, node_in, T]
            current node embedding for each of the n nodes.
        edge_index : torch.tensor, shape [2, E]
            indices of all edges
        edge_attr : torch.tensor, shape [E, edge_in, T]
            edge_attributes of each of the E edges
        r : torch.tensor, shape [E, dim, T]
            edge vectors
        L_init : torch.tensor, shape [E,]
            original distances between nodes
        node_type : torch.tensor, shape [n, 3]
            node type for each of the n nodes in a one-hot encoding.
        tau : torch.tensor, shape [n, 1]
            flow-matching time for each node, shape [n, 1].
        pos : torch.tensor, shape [n, dim, T]
            position of each node, shape [n, dim, T].
        batch : torch.tensor, shape [n,]
            batch information for each node, indicating which graph it belongs to.
        """
        tt = r.shape[-1]  # number of time steps

        # compute distance for each edge
        # Use sqrt(r^2 + eps) instead of norm for numerical stability in gradients
        eps = 1e-8
        d = torch.sqrt(torch.sum(r**2, dim=1, keepdim=True) + eps)  # shape [E, 1, T]

        if torch.isnan(d).any():
            raise ValueError('NaN values encountered in edge distances d during EdgeNodeUpdate forward pass.')

        # compute strain, relative distance and cos(q) for each edge
        # q: angle between edge vector and the y-axis
        cos_q = r[:, [1]]/d  # angle between edge vector and y-axis, shape [E, 1, T]
        L_init = L_init.reshape(-1, 1, 1)  # shape [E, 1, 1] to match d shape
        strain = (d - L_init)/(L_init + eps)    # shape [E, 1, T]
        d_rel, d_avg = self.relative_distance_calculator(edge_index, d)  # shape [E, 1, T]
        physical_quantity = torch.cat((strain, d_rel, cos_q, d, L_init.expand(-1, -1, tt)), dim=1) # shape [E, 5, T]

        if torch.isnan(physical_quantity).any():
            raise ValueError('NaN values encountered in physical_quantity.')

        # for name, value in zip(['edge_index', 'x', 'edge_attr', 'physical_quantity'], [edge_index, x, edge_attr, physical_quantity]):
        #     print(f'{name:15} {repr(value.shape):30} {value.dtype}')

        cos_phi, sin_phi = cos_sin_phi_from_pos(pos, node_type, batch)

        if torch.isnan(cos_phi).any():
            raise ValueError(f'NaN values encountered in cos_phi during EdgeNodeUpdate forward pass.')

        # node stuff:
        # x (= node_attr)
        # tau
        # cos(phi)

        # edge stuff:
        # physical_quantity (= strain, relative distance, cos(q))

        # time step stuff (change in node):
        # |dx|
        # x_i
        # tau (flow-matching time)
        # cos(dphi) = cos(q_i^(t+1) - q_i^(t))
        # cos(phi^t), cos(phi^(t+1))

        # # Time step embeddings
        # dx_norm = torch.norm(torch.diff(pos, dim=-1), dim=1, keepdim=True)  # shape [n, 1, T-1] # distance moved by each node in each time step
        # # dx_norm /= (0.5*d_avg[..., 1:] + 0.5*d_avg[..., :-1])
        # x_i = x.unsqueeze(2).expand(-1, -1, tt-1)  # shape [n, node_in + 1, T-1]
        # cos_dphi = cos_phi[..., 1:] * cos_phi[..., :-1] + sin_phi[..., 1:] * sin_phi[...,:-1]  # shape [n, 1, T-1]
        # dt_attr = torch.cat((dx_norm,
        #                      x_i,
        #                      tau.unsqueeze(2).expand(-1,-1,tt-1),
        #                      cos_dphi,
        #                      cos_phi[..., 1:],
        #                      cos_phi[..., :-1]), dim=1)  # shape [n, node_in + 6, T-1]

        # coeffs = self.dt_unet(tau, dt_attr)  # shape [n, 2, T-1]
        # dx = torch.diff(pos, dim=-1)  # shape [n, 2, T-1]
        # pos[..., 1:] += 0.2*(coeffs[:, 0:1])*dx/(1+dx_norm)  # update node positions based on previous time step
        # pos[..., :-1] += 0.2*(coeffs[:, 1:])*dx/(1+dx_norm)  # update node positions based on next time step

        x = torch.cat((x, cos_phi), dim=1)  # add cos_phi to node embedding # shape [n, node_in + 1, T]

        if torch.isnan(x).any():
            raise ValueError('NaN values encountered in node embeddings x after adding cos_phi during EdgeNodeUpdate forward pass.')

        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, physical_quantity=physical_quantity, tau=tau[edge_index[0]])

        if torch.isnan(x).any():
            raise ValueError('NaN values encountered in updated node embeddings x after message passing during EdgeNodeUpdate forward pass.')

        if not self.unet_edge is None:
            edge_attr = self.edge_updater(edge_index, tau=tau, messages=self.messages)
        else:
            edge_attr = torch.tensor([1.0], dtype=torch.float32,
                                   device=x.device)

        if torch.isnan(edge_attr).any():
            raise ValueError('NaN values encountered in updated edge embeddings edge_attr during EdgeNodeUpdate forward pass.')

        return x, edge_attr, self.messages, strain, pos, d

    def message(self, x_i, x_j, physical_quantity, edge_attr, tau):
        """computes message that each edge sends

        parameters
        ----------
        x_i : torch.tensor, shape [E, node_in, T]
            node embeddings of target nodes
        x_j : torch.tensor, shape [E, node_in, T]
            node embeddings of source nodes
        physical_quantity : torch.tensor, shape [E, p, T]
            p physical quantities associated with the edge between node j and node i (can be distance, reference distance, relative distance or strain)
        edge_attr : torch.tensor, shape [E, edge_in, T]
            edge embeddings
        """
        tt = x_i.shape[-1]  # number of time steps
        temp = torch.cat((x_j, x_i, physical_quantity, edge_attr), dim=1)
        if torch.isnan(temp).any():
            raise ValueError('NaN values encountered in input to unet_message during EdgeNodeUpdate message computation.')
        temp = self.unet_message(tau, temp)

        if torch.isnan(temp).any():
            raise ValueError('NaN values encountered in messages during EdgeNodeUpdate message computation.')

        return temp

    def update(self, aggr_out, x, tau):
        if self.unet_update is None:
            return torch.tensor([1.0], dtype=torch.float32, device=x.device)
        else:
            temp = torch.cat((x, aggr_out), dim=1)
            return self.unet_update(tau, temp)

    def edge_update(self, tau_i, messages):
        """_summary_

        Parameters
        ----------
        tau_i : torch.tensor, shape [e, 1]
            flow-matching time for each edge
        messages : None or torch.tensor, shape [e, message_size]
            message that each edge sends

        Returns
        -------
        torch.tensor, shape [e, edge_out]
            updated embedding of the edges
        """
        return self.unet_edge(tau_i, messages)

class PosUpdate(tg.nn.MessagePassing):
    """message passing layer that updates: the positions of the nodes and the vectors r_ij pointing along each edge.
    The shift in node position is calculated from a contribution of each incoming edge, this contribution depends on the message of this edge and the edge vector (which points from the location of the source node to the location of the target node).

    parameters
    ----------
    messagepassing : [type]
        [description]
    """

    def __init__(self, message_size):

        """initialize layer

        parameters
        ----------
        message_size : int
            size of the message
        """
        super().__init__(aggr='mean', node_dim=0)
        # self.mlp_shift = torch.nn.Sequential(
        #                 torch.nn.Linear(message_size, 1),
        #                 )
        # self.unet_shift = torch.nn.Sequential(
        #                 torch.nn.Linear(message_size, 1),
        #                 )
        self.unet_shift = UNet1D(message_size, 1, hidden_channels=message_size)

    def forward(self, edge_index, pos, r_norm, messages, strain, tau, verbose=False):
        """ From the messages, calculate updates to the node positions. N nodes, G graphs, E edges, T time steps, dim dimensions (always 2 for now).

        parameters
        ----------
        edge_index : torch.tensor, shape [2, E]
            indices of all edges
        r_norm : torch.tensor, shape [N, dim, T]
            normalized edge vectors
        pos : torch.tensor, shape [N, dim, T]
            current position of each node
        messages: torch.tensor, shape [E, message_size, T]
            messages sent by each edge
        strain: torch.tensor, shape [E, 1, T]
            strain of each edge, calculated as (d - L_init)/L_init, where d is the current distance between the nodes and L_init is the original distance between the nodes.
        """

        shift = self.propagate(edge_index, pos=pos, r_norm=r_norm, messages=messages, tau=tau, strain=strain)  # shape [N, dim, T]
        pos += shift
        r_new = r_norm + shift[edge_index[1]] - shift[edge_index[0]]  # shape [E, dim, T]
        return pos, r_new

    def message(self, r_norm, messages, tau, strain):
        # messages: shape [E, message_size, T]
        # r: shape [E, dim, T]
        # strain: shape [E, 1, T]
        # return shape [E, dim, T]
        # return r*torch.nn.Tanh()(strain*self.unet_shift(messages))
        # return r*(strain*self.unet_shift(messages))
        return r_norm*self.unet_shift(tau, messages)

    def update(self, aggr_out, pos):
        return aggr_out

class RelativeDistance(tg.nn.MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, d):
        """E edges, T time steps.

        parameters
        ----------
        edge_index : torch.tensor, shape [2, E]
            indices of all edges
        d : torch.tensor, shape [E, 1, T]
            edge distances
        """
        avg_d = self.propagate(edge_index, d=d.squeeze(1)).unsqueeze(1)  # shape [N, 1, T]
        eps = 1e-8
        rel_d = d/(avg_d[edge_index[1]] + eps)  # shape [E, 1, T]

        return rel_d, avg_d

    def message(self, d):
        return d

    def update(self, aggr_out):
        return aggr_out

def qe_to_pos(q, e, N, L_init, node_type, log_strain=True):
    """"Convert the quantities q and e to positions of the nodes. n nodes in total, N per beam, T time steps, G graphs/beams, S segments in total.

    Parameters
    ----------
    q : torch.tensor, shape [..., S, T]
        angles of the segments, where n is the number of nodes and T is the number of time steps.
    e : torch.tensor, shape [..., S, T]
        strains of the segments
    N : torch.tensor, shape [G,]
        number of segments in each beam
    L_init : torch.tensor, shape [S,]
        initial lengths of the segments
    node_type : torch.tensor, shape [n, 3]
        node type for each of the n nodes in a one-hot encoding
    log_strain : bool, optional
        if True assume logarithmic strain (Hencky strain)
        else assume engineering strain (Cauchy strain)
        (default is True)

    Returns
    -------
    pos : torch.tensor, shape [..., n_nodes, 2, T]
        positions of the nodes over time
    """
    S = q.shape[-2]  # number of segments in total
    T = q.shape[-1]  # number of time steps
    G = N.shape[0]  # number of beams
    n_nodes = torch.sum(N+1)  # total number of nodes in all beams

    L = L_init.unsqueeze(-1) * torch.exp(e) if log_strain else L_init.unsqueeze(-1) + e * L_init.unsqueeze(-1)  # shape [..., S, T]

    shape = list(q.shape)
    shape = shape[:-1] + [2, shape[-1]]  # add 1 dimension of size 2 for x and y coordinates
    shape[-3] = n_nodes  # number of nodes in total

    pos = torch.empty(shape, device=q.device, dtype=q.dtype)  # initialize positions

    ind_start = torch.empty_like(N, device=N.device, dtype=torch.long)  # indices of the start node in each beam, shape [G,]
    ind_start[0] = 0  # first beam starts at index 0
    ind_start[1:] = torch.cumsum(N + 1, dim=0)[:-1]  # indices of the start node in each beam, shape [G,]

    pos[..., node_type[:, 1] != 1, 0, :] = batched_cumsum(-L*torch.sin(q), N, dim=q.ndim-2)  # x-coordinates of all nodes except the start nodes
    pos[..., node_type[:, 1] != 1, 1, :] = batched_cumsum(L*torch.cos(q), N, dim=q.ndim-2)  # y-coordinates of all nodes except the start nodes
    pos[..., node_type[:, 1] == 1, :, :] = 0  # coordinates of the first node

    return pos

def batched_cumsum(x, N, dim=0):
    """Cumulative sum for each graph in x, assuming the graphs are concatenated along the graph dimension. G graphs, N elements in each.

    Parameters
    -----------
    x : torch.tensor, any shape where dimension dim has size sum(`N`).
        Tensor to perform cumulative sum on.
    N : torch.tensor, shape [G,]
       number of elements in each graph.
    dim: int, optional
        The dimension along which to perform the cumulative sum (by default 0).

    Returns
    --------
    torch.tensor, shape same as `x`
        `x with the cumulative sum applied along the specified dimension, such that each graph's cumulative sum is calculated separately.

    Examples
    --------
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> N = torch.tensor([2, 1])
    >>> batched_cumsum(x, N, dim=0)
    tensor([[ 1,  2,  3],
            [ 5,  7,  9],
            [ 7,  8,  9]])
    """

    if dim > x.ndim - 1 or dim < -1:
        raise ValueError(f"Invalid dimension {dim} for tensor with {x.ndim} dimensions.")
    if torch.sum(N) != x.shape[dim]:
        raise ValueError(f"Sum of N ({torch.sum(N)}) must match the size of dimension {dim} in x ({x.shape[dim]}).")

    # first, perform regular cumulative sum
    x = torch.cumsum(x, dim=dim)

    if len(N) > 1:
        # However, this treats the entire tensor as one long sequence, so we need to reset the cumulative sum at the start of each graph.
        # Subtract cumsum of previous graph from the next graph.
        end_inds = torch.cumsum(N, dim=0)[:-1]-1  # end indices of each graph (except last), shape [G-1,]

        x_end = x.index_select(dim, end_inds)  # get the last element of each graph, shape [..., G-1, ...] (same as shape of x with only dimension dim changed)
        x_end = torch.repeat_interleave(x_end, repeats=N[1:], dim=dim)  # repeat each end element according to the number of elements in the next graph, shape [..., n-N[0], ...]

        shape = [1] * x.ndim
        shape[dim] = -1
        end_inds = end_inds.reshape(shape)  # reshape end_inds to match the shape of x for broadcasting, shape [..., G-1, ...]

        len_x = x.shape[dim]  # n
        x_narrow = torch.narrow(x, dim, N[0], len_x-N[0]) # take x[..., N[0]:, ...], resulting in shape [..., n-N[0], ...]
        x_narrow -= x_end

    return x

class Prior():
    def __init__(self, lamb, mu_eps, std_eps):
        self.lamb = lamb
        self.expdist = torch.distributions.Exponential(lamb)
        self.mu_eps = mu_eps
        self.std_eps = std_eps

    def __call__(self, N, d, node_type, L_init, batch, L_tot, n_samples=1):
        """Generate prior samples for the model. G graphs, N segments in each beam, T time steps, S segments in total, n nodes in total.

        Parameters
        ----------
        N : torch.tensor, shape [G]
            number of segments in each beam
        d : torch.tensor, shape [G, T]
            tip displacement of each beam over time
        node_type : torch.tensor, shape [n, 3]
            node type for each of the n nodes in a one-hot encoding
        L_init : torch.tensor, shape [S,]
            initial lengths of the segments
        batch : torch.tensor, shape [n,]
            batch indices for each node, shape [n, 1]. This is used to select the correct beam for each node.
        L_tot : torch.tensor, shape [G,]
            total lengths of each beam. This is used to fix the y-coordinate of the end node.
        n_samples : int
            number of samples to generate per beam, default 1

        Returns
        -------
        pos : torch.tensor, shape [n_samples, n, 2, T] or [n, 2, T] if n_samples=1
            prior samples of the positions of each node for each time step.
            The y-coordinate of the end node is fixed to L_tot-d, while the x and y-coordinate of the start node are fixed to 0.
        """
        tt = d.shape[-1]  # number of time steps
        S = L_init.shape[0]  # number of segments in total
        N_b = torch.repeat_interleave(N, N, dim=0)  # shape [S,]

        # sample phi (shape [n_samples, S, tt])
        # -------------------------------------------------
        # sample random noise z ~ N(0, std_phi)
        q_sample = self.expdist.sample((n_samples, S, tt)).to(device=N.device)

        # scale random noise
        q_sample /= N_b.unsqueeze(-1)

        # set angle at t=0 to 0
        q_sample[:, :, 0] = 0.0

        # take cumsum over time, such that phi^t = phi^{t-1} + z
        q_sample = -torch.cumsum(q_sample, dim=2)

        # take cumsum of phi over segments to get q
        q_sample = batched_cumsum(q_sample, N, dim=1)

        # sample eps (shape [n_samples, S, tt])
        # -------------------------------------------------
        # sample random noise z ~ N(mu_std, std_eps)
        eps_sample = torch.randn(n_samples, S, tt, device=N.device) * self.std_eps + self.mu_eps

        # scale random noise, taking into account average random walk length
        eps_sample /= N_b.unsqueeze(-1)

        # set eps at t=0 to 0
        eps_sample[:, :, 0] = 0.0

        # take cumsum over time, such that eps^t = eps^{t-1} + z
        eps_sample = torch.cumsum(eps_sample, dim=2)

        # convert q, eps to positions of nodes, shape [n_samples, n, 2, T]
        pos = qe_to_pos(q_sample, eps_sample, N, L_init, node_type)

        plusminus = torch.sign(torch.rand(batch[-1] + 1, device=N.device) - 0.5)  # random sign for each beam, shape [G,]
        plusminus = plusminus[batch]  # shape [n,]
        pos[:, :, 0, :] *= plusminus.unsqueeze(-1)  # flip x-coordinates of all nodes in each beam randomly

        # shift nodes such that end node is at y = L_tot - d
        y_N_current = pos[:, node_type[:, 2] == 1, 1][:, batch]  # shape [n_samples, n, T]
        y_N_goal = (L_tot.unsqueeze(-1) - d)[batch]  # shape [n, T]

        ind = batched_cumsum(torch.ones(pos.shape[1], device=pos.device),
                             N+1, dim=0)-1  # 0, 1, 2, etc. for each beam, shape [n, 1]

        pos[:, :, 1, :] -= ind.unsqueeze(-1)/N[batch].unsqueeze(-1)*(y_N_current - y_N_goal)

        if n_samples == 1:
            return pos.squeeze(0)
        else:
            return pos


class Prior_wide():
    def __init__(self, std_phi, mu_eps, std_eps, a, b):
        self.std_phi = std_phi
        self.mu_eps = mu_eps
        self.std_eps = std_eps
        self.a = a
        self.b = b

    def __call__(self, N, d, node_type, L_init, batch, L_tot, n_samples=1):
        """Generate prior samples for the model. G graphs, N segments in each beam, T time steps, S segments in total, n nodes in total.

        Parameters
        ----------
        N : torch.tensor, shape [G]
            number of segments in each beam
        d : torch.tensor, shape [G, T]
            tip displacement of each beam over time
        node_type : torch.tensor, shape [n, 3]
            node type for each of the n nodes in a one-hot encoding
        L_init : torch.tensor, shape [S, 1]
            initial lengths of the segments
        batch : torch.tensor, shape [n,]
            batch indices for each node, shape [n, 1]. This is used to select the correct beam for each node.
        L_tot : torch.tensor, shape [G, 1]
            total lengths of each beam. This is used to fix the y-coordinate of the end node.
        n_samples : int
            number of samples to generate per beam, default 1

        Returns
        -------
        pos : torch.tensor, shape [n_samples, n, 2, T] or [n, 2, T] if n_samples=1
            prior samples of the positions of each node for each time step.
            The y-coordinate of the end node is fixed to L_tot-d, while the x and y-coordinate of the start node are fixed to 0.
        """
        tt = d.shape[-1]  # number of time steps
        S = L_init.shape[0]  # number of segments in total
        N_b = torch.repeat_interleave(N, N, dim=0)  # shape [S,]

        # sample phi (shape [n_samples, S, tt])
        # -------------------------------------------------
        # sample random noise z ~ N(0, std_phi)
        q_sample = torch.randn(n_samples, S, tt, device=N.device) * self.std_phi

        # scale random noise, taking into account average random walk length
        q_sample *= np.sqrt(tt*np.pi/2) / N_b.unsqueeze(-1)

        # set angle at t=0 to 0
        q_sample[:, :, 0] = 0.0

        # take cumsum over time, such that phi^t = phi^{t-1} + z
        q_sample = torch.cumsum(q_sample, dim=2)

        # take cumsum of phi over segments to get q
        q_sample = batched_cumsum(q_sample, N, dim=1)

        # sample eps (shape [n_samples, S, tt])
        # -------------------------------------------------
        # sample random noise z ~ N(mu_std, std_eps)
        eps_sample = torch.randn(n_samples, S, tt, device=N.device) * self.std_eps + self.mu_eps

        # scale random noise, taking into account average random walk length
        eps_sample *= np.sqrt(tt*np.pi/2) / N_b.unsqueeze(-1)

        # set eps at t=0 to 0
        eps_sample[:, :, 0] = 0.0

        # take cumsum over time, such that eps^t = eps^{t-1} + z
        eps_sample = torch.cumsum(eps_sample, dim=2)

        # convert q, eps to positions of nodes, shape [n_samples, n, 2, T]
        pos = qe_to_pos(q_sample, eps_sample, N, L_init, node_type)

        # shift nodes such that end node is at y = L_tot - d
        y_N_current = pos[:, node_type[:, 2] == 1, 1][:, batch]  # shape [n_samples, n, T]
        y_N_goal = (L_tot.unsqueeze(-1) - d)[batch]  # shape [n, T]

        ind = batched_cumsum(torch.ones(pos.shape[1], device=pos.device),
                             N+1, dim=0)-1  # 0, 1, 2, etc. for each beam, shape [n, 1]

        pos[:, :, 1, :] -= ind.unsqueeze(-1)/N[batch].unsqueeze(-1)*(y_N_current - y_N_goal)

        if n_samples == 1:
            return pos.squeeze(0)
        else:
            return pos


class EGNN(torch.nn.Module):
    def __init__(self, layers, reuse_layers=None):
        """Initialize an E(N) equivariant GNN for beam.

        Parameters
        ----------
        layers : tuple of tuples
            each element of the tuple is itself a tuple with 5 elements, which are: node_in, edge_in, message_size, node_out, edge_out
        reuse_layers : tuple of ints, optional
            how often each layer should be applied. If None, each layer will be applied once. By default None
        """

        super().__init__()

        if reuse_layers is not None:
            if len(layers) != len(reuse_layers):
                raise ValueError(f'length of layers (currently {len(layers)}) must be equal to length of reuse_layers (currently {len(reuse_layers)}')
        #     if not reuse_layers == (1,):
        #         raise NotImplementedError(f'Only one layer with no re-use supported for now, but got reuse_layers={reuse_layers}')

        # if len(layers) != 1:
        #     raise NotImplementedError(f'Only one layer supported for now, but got layers={layers}')

        self.num_layers = len(layers)

        self.edgeNodeUpdates = torch.nn.ModuleList(
                [EdgeNodeUpdate(*layer) for layer in layers]
            )
        self.posUpdates = torch.nn.ModuleList(
                [PosUpdate(layer[2]) for layer in layers]
            )

        if reuse_layers is None:
            self.reuse_layers = (1,) * len(layers)
        else:
            self.reuse_layers = reuse_layers

        self.num_mpsteps = sum(self.reuse_layers)


    def forward(self, batch, current_pos, tau, verbose=False):
        """n nodes, G graphs, E edges, T time steps, N segments per beam. y-coordinate of the tip node is already assumed to be at L_tot-d in the input current_pos, and this will also be the case in the output.

        Parameters
        ----------
        batch : pytorch_geometric.data.Data
            batch of graphs, containing the following attributes:
            edge_index : torch.tensor, shape [2, E]
                indices of all edges (assumed constant in time)
            node_attr : torch.tensor, shape [n, node_in]
                node attributes for each of the n nodes
            edge_attr : torch.tensor, shape [E, edge_in, T]
                edge attributes for each of the E edges
            batch : torch.tensor, shape [n, 1]
                batch indices for each node
            L_init : torch.tensor, shape [E, 1]
                original distances between nodes
            L_tot : torch.tensor, shape [G, 1]
                total lengths of each beam
            d : torch.tensor, shape [G, T]
                tip displacement of each beam over time
            N : torch.tensor, shape [G, 1]
                number of segments in each beam
        current_pos : torch.tensor, shape [n, 2, T]
            current position of each node
        tau : float or torch.tensor, shape [G,]
            current flow-matching time (not the same as trajectory time t!)
            If float, it is assumed to be the same for all beams.
            If torch.tensor, it is assumed to be different for each beam.
        verbose : bool, optional
            whether to print verbose output, by default False

        Returns
        -------
        shift : torch.tensor, shape [n, 2, T]
            shift in positions of each node for each time step.
            pos_predicted = pos + shift
            The y-coordinate of the end node is fixed to L_tot-d, while the x and y-coordinate of the start node are fixed to 0.
        """
        if verbose:
            print('======== Starting EGNN forward pass =======')

        edge_index, node_attr, edge_attr, batch2, L_init, L_tot, d, N = batch.edge_index, batch.node_attr, batch.edge_attr, batch.batch, batch.L_init, batch.L_tot, batch.d, batch.N

        L_tot = L_tot.unsqueeze(-1)

        node_type = node_attr[:, :3].clone()

        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor([tau]*node_attr.shape[0], device=node_attr.device, dtype=node_attr.dtype).unsqueeze(-1)  # shape [n, 1]
        else:
            tau = tau.unsqueeze(-1)  # shape [G, 1]
            tau = tau[batch2]  # shape [n, 1] to match node_attr
        edge_attr = torch.cat((edge_attr, tau[edge_index[0]]), dim=1)  # add tau to edge attributes, shape [E, edge_in + 1]

        # clone pos to prevent modifying the input
        pos = current_pos.clone()

        r = pos[edge_index[1]] - pos[edge_index[0]]  # edge vectors [E, 2, T]
        r_init = r.clone()

        if verbose:
            print('initial guess pos:')
            print(pos)

        tt = current_pos.shape[-1]  # number of time steps

        node_attr = node_attr.unsqueeze(2).expand(-1, -1, tt)  # shape [n, node_in, T]
        edge_attr = edge_attr.unsqueeze(2).expand(-1, -1, tt)  # shape [E, edge_in, T]

        if verbose:
            for name, value in [('node_attr', node_attr),
                                ('edge_attr', edge_attr),
                                ('pos', pos),
                                ('r', r),
                                ]:
                print(f'{name:15} mean: {value.mean():<.3f}, std: {value.std():<.3f} min: {value.min():<.3f}, max: {value.max():<.3f}')

        for i in range(self.num_layers):
            for j in range(self.reuse_layers[i]):
                if verbose:
                    print(f'-------- EGNN layer {i}, repetition {j} --------')

                node_attr, edge_attr, messages, strain, pos, L = self.edgeNodeUpdates[i](node_attr, edge_index, edge_attr=edge_attr, r=r, L_init=L_init, node_type=node_type, tau=tau, pos=pos, batch=batch2, verbose=verbose)

                if torch.isnan(node_attr).any():
                    raise ValueError(f'NaN values encountered in node_attr during GNN forward pass, layer {i}, repetition {j}.')
                if torch.isnan(edge_attr).any():
                    raise ValueError(f'NaN values encountered in edge_attr during GNN forward pass, layer {i}, repetition {j}.')
                if torch.isnan(messages).any():
                    raise ValueError(f'NaN values encountered in messages during GNN forward pass, layer {i}, repetition {j}.')
                if torch.isnan(strain).any():
                    raise ValueError(f'NaN values encountered in strain during GNN forward pass, layer {i}, repetition {j}.')
                if torch.isnan(L).any():
                    raise ValueError(f'NaN values encountered in L during GNN forward pass, layer {i}, repetition {j}.')

                eps = 1e-8
                pos, r = self.posUpdates[i](edge_index, pos, r_init/(L+eps), messages, strain, tau, verbose=verbose)

                if torch.isnan(pos).any():
                    raise ValueError(f'NaN values encountered in pos during GNN forward pass, layer {i}, repetition {j}.')
                if torch.isnan(r).any():
                    raise ValueError(f'NaN values encountered in r during GNN forward pass, layer {i}, repetition {j}.')

                # enforce boundary conditions
                # 1) keep first time step fixed (to do: shift other time steps as well)
                pos[..., 0] = current_pos[..., 0]
                # 2) shift all nodes such that the start node is at (0,0)
                pos -= pos[node_type[:, 1] == 1][batch2]

                y_N_current = pos[node_type[:, 2] == 1, 1][batch2]  # shape [n, T]
                y_N_goal = (L_tot - d)[batch2]  # shape [n, T]

                # 3) shift all nodes such that the end node is at y = L_tot - d
                # to do: use cumulative length here instead of ind.
                ind = batched_cumsum(torch.ones(pos.shape[0], device=pos.device), N.squeeze(-1)+1, dim=0)-1  # 0, 1, 2, etc. for each beam, shape [n, 1]
                pos[:, 1] -= (ind/N[batch2]).unsqueeze(-1)*(y_N_current - y_N_goal)

                if torch.isnan(pos).any():
                    raise ValueError(f'NaN values encountered in pos during GNN forward pass, layer {i}, repetition {j}.')

                if verbose:
                    for name, value in [('node_attr', node_attr),
                                        ('edge_attr', edge_attr),
                                        ('messages', messages),
                                        ('strain', strain),
                                        ('pos', pos),
                                        ('r', r),
                                        ('L', L)]:
                        print(f'{name:15} mean: {value.mean():<.3f}, std: {value.std():<.3f} min: {value.min():<.3f}, max: {value.max():<.3f}')

                    # print(f'{"ind.shape:":<20} {ind.shape}')
                    # print(f'{"pos.shape:":<20} {pos.shape}')
                    # print(f'{"N[batch].shape:":<20} {N[batch2].shape}')
                    # print(f'{"y_N_current.shape:":<20} {y_N_current.shape}')
                    # print(f'{"y_N_goal.shape:":<20} {y_N_goal.shape}')

                if verbose:
                    print(f'predicted pos[:3, :, :2] after fixing begin and end node, MP layer {i}, repetition {j}:')
                    print(pos[:3, :, :2])

        return (pos - current_pos)  #/(1-tau.unsqueeze(-1))

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    UNet = UNet1D(
        in_channels=15,
        out_channels=1,
        hidden_channels=32,
    ).to(device)

    temptemp = torch.rand(13, 15, 123).to(device)
    print(UNet(0, temptemp, verbose=True).shape)

    print('\n\n')

    temptemp = torch.rand(7, 15, 200).to(device)
    print(UNet(0, temptemp, verbose=True).shape)

    # pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #                     [[0.0, 0.0, 0.1], [0.5, 0.4, 0.35]],
    #                     [[0.0, 0.0, 0.2], [0.75, 0.6, 0.55]],
    #                     [[0.0, 0.0, 0.3], [1.0, 0.7, 0.6]]])
    # node_type = torch.tensor([[0, 1, 0],
    #                           [1, 0, 0],
    #                           [1, 0, 0],
    #                           [0, 0, 1]], device=pos.device)

    # cos_phi, sin_phi = cos_sin_phi_from_pos(pos, node_type)
    # print("cos_phi:", cos_phi)
    # print("sin_phi:", sin_phi)

    # pos = torch.tensor([[[0.0], [0.0]],
    #                     [[1.0], [1.0]],
    #                     [[2.0], [0.0]]])
    # node_type = torch.tensor([[0, 1, 0],
    #                           [1, 0, 0],
    #                           [0, 0, 1]], device=pos.device)

    # cos_phi, sin_phi = cos_sin_phi_from_pos(pos, node_type)
    # print("\ncos_phi:", cos_phi)
    # print("sin_phi:", sin_phi)

    # pos = torch.tensor([[[0.0], [0.0]],
    #                     [[1.0], [1.0]],
    #                     [[2.0], [0.0]],
    #                     [[0.0], [0.0]],
    #                     [[1.0], [1.0]],
    #                     [[2.0], [0.0]]])
    # node_type = torch.tensor([[0, 1, 0],
    #                           [1, 0, 0],
    #                           [0, 0, 1],
    #                           [0, 1, 0],
    #                           [1, 0, 0],
    #                           [0, 0, 1]], device=pos.device)
    # batch = torch.tensor([0, 0, 0, 1, 1, 1], device=pos.device)

    # cos_phi, sin_phi = cos_sin_phi_from_pos(pos, node_type, batch=batch)
    # print("\ncos_phi:", cos_phi)
    # print("sin_phi:", sin_phi)
