import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
import torchac
import math
import multiprocessing

import _gridencoder as _backend

anchor_round_digits = 16
Q_anchor = 1/(2 ** anchor_round_digits - 1)
use_clamp = True
use_multiprocessor = False  # Always False plz. Not yet implemented for True.

def get_binary_vxl_size(binary_vxl):
    # binary_vxl: {0, 1}
    # assert torch.unique(binary_vxl).mean() == 0.5
    ttl_num = binary_vxl.numel()

    pos_num = torch.sum(binary_vxl)
    neg_num = ttl_num - pos_num

    Pg = pos_num / ttl_num  #  + 1e-6
    Pg = torch.clamp(Pg, min=1e-6, max=1-1e-6)
    pos_prob = Pg
    neg_prob = (1 - Pg)
    pos_bit = pos_num * (-torch.log2(pos_prob))
    neg_bit = neg_num * (-torch.log2(neg_prob))
    ttl_bit = pos_bit + neg_bit
    ttl_bit += 32  # Pg
    # print('binary_vxl:', Pg.item(), ttl_bit.item(), ttl_num, pos_num.item(), neg_num.item())
    return Pg, ttl_bit, ttl_bit.item()/8.0/1024/1024, ttl_num


def multiprocess_encoder(lower, symbol, file_name, chunk_num=10):
    def enc_func(l, s, f, b_l, i):
        byte_stream = torchac.encode_float_cdf(l, s, check_input_bounds=True)
        with open(f, 'wb') as fout:
            fout.write(byte_stream)
        bit_len = len(byte_stream) * 8
        b_l[i] = bit_len
    encoding_len = lower.shape[0]
    chunk_len = int(math.ceil(encoding_len / chunk_num))
    processes = []
    manager = multiprocessing.Manager()
    b_list = manager.list([None] * chunk_num)
    for m_id in range(chunk_num):
        lower_m = lower[m_id * chunk_len:(m_id + 1) * chunk_len]
        symbol_m = symbol[m_id * chunk_len:(m_id + 1) * chunk_len]
        file_name_m = file_name.replace('.b', f'_{m_id}.b')
        process = multiprocessing.Process(target=enc_func, args=(lower_m, symbol_m, file_name_m, b_list, m_id))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    ttl_bit_len = sum(list(b_list))
    return ttl_bit_len


def multiprocess_deoder(lower, file_name, chunk_num=10):
    def dec_func(l, f, o_l, i):
        with open(f, 'rb') as fin:
            byte_stream_d = fin.read()
        o = torchac.decode_float_cdf(l, byte_stream_d).to(torch.float32)
        o_l[i] = o
    encoding_len = lower.shape[0]
    chunk_len = int(math.ceil(encoding_len / chunk_num))
    processes = []
    manager = multiprocessing.Manager()
    output_list = manager.list([None] * chunk_num)
    for m_id in range(chunk_num):
        lower_m = lower[m_id * chunk_len:(m_id + 1) * chunk_len]
        file_name_m = file_name.replace('.b', f'_{m_id}.b')
        process = multiprocessing.Process(target=dec_func, args=(lower_m, file_name_m, output_list, m_id))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    output_list = torch.cat(list(output_list), dim=0).cuda()
    return output_list


def encoder_gaussian(x, mean, scale, Q, file_name='tmp.b'):
    assert file_name.endswith('.b')
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert x.shape == mean.shape == scale.shape == Q.shape
    x_int_round = torch.round(x / Q)  # [100]
    max_value = x_int_round.max()
    min_value = x_int_round.min()
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1 + 1)).to(
        torch.float).to(x.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]
    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    del samples
    del mean
    del scale
    del GD
    x_int_round_idx = (x_int_round - min_value).to(torch.int16)
    assert (x_int_round_idx.to(torch.int32) == x_int_round - min_value).all()
    # if x_int_round_idx.max() >= lower.shape[-1] - 1:  x_int_round_idx.max() exceed 65536 but to int6, that's why error
        # assert False

    if not use_multiprocessor:
        byte_stream = torchac.encode_float_cdf(lower.cpu(), x_int_round_idx.cpu(), check_input_bounds=True)
        with open(file_name, 'wb') as fout:
            fout.write(byte_stream)
        bit_len = len(byte_stream)*8
    else:
        bit_len = multiprocess_encoder(lower.cpu(), x_int_round_idx.cpu(), file_name)
    torch.cuda.empty_cache()
    return bit_len, min_value, max_value


def decoder_gaussian(mean, scale, Q, file_name='tmp.b', min_value=-100, max_value=100):
    assert file_name.endswith('.b')
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert mean.shape == scale.shape == Q.shape
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1 + 1)).to(
        torch.float).to(mean.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]
    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    if not use_multiprocessor:
        with open(file_name, 'rb') as fin:
            byte_stream_d = fin.read()
        sym_out = torchac.decode_float_cdf(lower.cpu(), byte_stream_d).to(mean.device).to(torch.float32)
    else:
        sym_out = multiprocess_deoder(lower.cpu(), file_name, chunk_num=10).to(torch.float32)
    x = sym_out + min_value
    x = x * Q
    torch.cuda.empty_cache()
    return x


def encoder(x, p, file_name):
    x = x.detach().cpu()
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    sym = torch.floor(((x+1)/2)).to(torch.int16)
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
    # Number of bits taken by the stream
    bit_len = len(byte_stream) * 8
    # Write to a file.
    with open(file_name, 'wb') as fout:
        fout.write(byte_stream)
    return bit_len

def decoder(p, file_name):
    dvc = p.device
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    # Read from a file.
    with open(file_name, 'rb') as fin:
        byte_stream = fin.read()
    # Decode from bytestream.
    sym_out = torchac.decode_float_cdf(output_cdf, byte_stream)
    sym_out = (sym_out * 2 - 1).to(torch.float32)
    return sym_out.to(dvc)


def encoder_anchor(anchors, anchor_digits=anchor_round_digits, file_name='anchor.b'):
    anchors = anchors.detach()
    anchors_voxel = anchors / Q_anchor
    anchor_voxel_min = (torch.min(anchors_voxel, dim=0, keepdim=True)[0])
    anchor_voxel_max = (torch.max(anchors_voxel, dim=0, keepdim=True)[0])

    anchors_voxel -= anchor_voxel_min
    anchors_voxel = anchors_voxel.to(torch.long)
    voxel_hwd = anchor_voxel_max - anchor_voxel_min + 1

    voxel_hwd = voxel_hwd.to(torch.long).squeeze()
    voxel_1 = torch.zeros(size=voxel_hwd.tolist(), device='cuda', dtype=torch.long)

    anchors_unique_values, anchors_unique_cnts = torch.unique(anchors_voxel, dim=0, return_counts=True)
    # anchors_reassembled = torch.repeat_interleave(anchors_unique_values, anchors_unique_cnts.to(torch.long), dim=0)
    _, inv = torch.unique(anchors_voxel, dim=0, return_inverse=True)
    indices = torch.argsort(inv)

    anchors_unique_values = anchors_unique_values.to(torch.long)
    mask = anchors_unique_cnts == 1
    anchors_unique_values_1 = anchors_unique_values[mask]
    anchors_unique_values_not1 = anchors_unique_values[~mask]
    anchors_unique_cnts_not1 = anchors_unique_cnts[~mask]
    voxel_1[anchors_unique_values_1[:, 0], anchors_unique_values_1[:, 1], anchors_unique_values_1[:, 2]] = 1
    voxel_1 = voxel_1.view(-1)
    prob_1 = (voxel_1.sum() / voxel_1.numel()).item()
    p = torch.zeros_like(voxel_1).to(torch.float32)
    p[...] = prob_1
    voxel_1 = voxel_1 * 2 - 1

    file_name_value_not1 = file_name.replace('.b', '_value_not1.b')
    file_name_cnts_not1 = file_name.replace('.b', '_cnts_not1.b')

    # Directly store them for simplicity
    torch.save(anchors_unique_values_not1, f=file_name_value_not1)  # assume we use digit=anchor_digits to encode.
    torch.save(anchors_unique_cnts_not1, f=file_name_cnts_not1)  # assume we use digit=16 to encode.

    bit_len_anchor_1 = encoder(voxel_1, p, file_name)
    # decoded_list = decoder(p, file_name)  # {-1, 1}
    bit_len_anchor_not1 = anchor_digits * anchors_unique_values_not1.numel() + 16 * anchors_unique_cnts_not1.numel()
    bit_len_anchor = bit_len_anchor_1 + bit_len_anchor_not1
    return bit_len_anchor, indices, voxel_hwd, anchor_voxel_min, anchor_voxel_max, prob_1


def decoder_anchor(voxel_hwd, prob_1, anchor_voxel_min, anchor_voxel_max, file_name='anchor.b'):
    voxel_1 = torch.zeros(size=voxel_hwd.tolist(), device='cuda', dtype=torch.long)
    p = torch.zeros_like(voxel_1).to(torch.float32)
    p[...] = prob_1
    decoded_list = decoder(p, file_name)  # {-1, 1}. shape=[h, w, d]
    decoded_list = (decoded_list + 1) / 2  # {0, 1}
    decoded_voxel_1 = torch.argwhere(decoded_list == 1)

    file_name_value_not1 = file_name.replace('.b', '_value_not1.b')
    file_name_cnts_not1 = file_name.replace('.b', '_cnts_not1.b')
    decoded_voxel_value_not1 = torch.load(file_name_value_not1)
    decoded_voxel_cnts_not1 = torch.load(file_name_cnts_not1)
    decodedc_voxel_not1 = torch.repeat_interleave(decoded_voxel_value_not1, decoded_voxel_cnts_not1.to(torch.long), dim=0)

    decodedc_voxel = torch.cat([decoded_voxel_1, decodedc_voxel_not1], dim=0).to(torch.float32)

    v, c = torch.unique(decodedc_voxel, dim=0, return_counts=True)
    decodedc_reassembled = torch.repeat_interleave(v, c.to(torch.long), dim=0)

    decoded_anchor = (decodedc_reassembled + anchor_voxel_min) * Q_anchor

    return decoded_anchor


class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        # out = torch.sign(input)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # mask: to ensure x belongs to (-1, 1)
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q, input_mean=None):
        if use_clamp:
            if input_mean is None:
                input_mean = input.mean()
            input_min = input_mean - 15_000 * Q
            input_max = input_mean + 15_000 * Q
            input = torch.clamp(input, min=input_min.detach(), max=input_max.detach())

        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q
        return Q_q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Quantize_anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchors, min_v, max_v):
        # if anchor_round_digits == 32:
            # return anchors
        # min_v = torch.min(anchors).detach()
        # max_v = torch.max(anchors).detach()
        # scales = 2 ** anchor_round_digits - 1
        interval = ((max_v - min_v) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        # quantized_v = (anchors - min_v) // interval
        quantized_v = torch.div(anchors - min_v, interval, rounding_mode='floor')
        quantized_v = torch.clamp(quantized_v, 0, 2 ** anchor_round_digits - 1)
        anchors_q = quantized_v * interval + min_v
        return anchors_q, quantized_v
    @staticmethod
    def backward(ctx, grad_output, tmp):  # tmp is for quantized_v:)
        return grad_output, None, None


class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets_list, resolutions_list, calc_grad_inputs=False, min_level_id=None, n_levels_calc=1, binary_vxl=None, PV=0):
        # inputs: [N, num_dim], float in [0, 1]
        # embeddings: [sO, n_features], float. self.params = nn.Parameter(torch.empty(offset, n_features))
        # offsets_list: [n_levels + 1], int
        # RETURN: [N, F], float
        inputs = inputs.contiguous()
        # embeddings_mask = torch.ones(size=[embeddings.shape[0]], dtype=torch.bool, device='cuda')
        # print('kkkkkkkkkk---000000000:', embeddings_mask.shape, embeddings.shape, embeddings_mask.sum())

        Rb = 128
        if binary_vxl is not None:
            binary_vxl = binary_vxl.contiguous()
            Rb = binary_vxl.shape[-1]
            assert len(binary_vxl.shape) == inputs.shape[-1]

        N, num_dim = inputs.shape # batch size, coord dim # N_rays, 3
        n_levels = offsets_list.shape[0] - 1 # level # 层数=16
        n_features = embeddings.shape[1] # embedding dim for each level # 就是channel数=2

        max_level_id = min_level_id + n_levels_calc

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if n_features % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and n_features % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # n_levels first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(n_levels_calc, N, n_features, device=inputs.device, dtype=embeddings.dtype)  # 创建一个buffer给cuda填充
        # outputs = [hash层数=16, N_rays, channels=2]

        # zero init if we only calculate partial levels
        # if n_levels_calc < n_levels: outputs.zero_()
        if calc_grad_inputs:  # inputs.requires_grad
            dy_dx = torch.empty(N, n_levels_calc * num_dim * n_features, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        # assert embeddings.shape[0] == embeddings_mask.shape[0]
        # assert embeddings_mask.dtype == torch.bool

        if isinstance(min_level_id, int):
            _backend.grid_encode_forward(
                inputs,
                embeddings,
                # embeddings_mask,
                offsets_list[min_level_id:max_level_id+1],
                resolutions_list[min_level_id:max_level_id],
                outputs,
                N, num_dim, n_features, n_levels_calc, 0, Rb, PV,
                dy_dx,
                binary_vxl,
                None
                )
        else:
            _backend.grid_encode_forward(
                inputs,
                embeddings,
                # embeddings_mask,
                offsets_list,
                resolutions_list,
                outputs,
                N, num_dim, n_features, n_levels_calc, 0, Rb, PV,
                dy_dx,
                binary_vxl,
                min_level_id
                )

        # permute back to [N, n_levels * n_features]  # [N_rays, hash层数=16 * channels=2]
        outputs = outputs.permute(1, 0, 2).reshape(N, n_levels_calc * n_features)

        ctx.save_for_backward(inputs, embeddings, offsets_list, resolutions_list, dy_dx, binary_vxl)
        ctx.dims = [N, num_dim, n_features, n_levels_calc, min_level_id, max_level_id, Rb, PV]  # min_level_id是否要单独save为tensor

        return outputs

    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets_list, resolutions_list, dy_dx, binary_vxl = ctx.saved_tensors
        N, num_dim, n_features, n_levels_calc, min_level_id, max_level_id, Rb, PV = ctx.dims

        # grad: [N, n_levels * n_features] --> [n_levels, N, n_features]
        grad = grad.view(N, n_levels_calc, n_features).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        if isinstance(min_level_id, int):
            _backend.grid_encode_backward(
                grad,
                inputs,
                embeddings,
                # embeddings_mask,
                offsets_list[min_level_id:max_level_id+1],
                resolutions_list[min_level_id:max_level_id],
                grad_embeddings,
                N, num_dim, n_features, n_levels_calc, 0, Rb,
                dy_dx,
                grad_inputs,
                binary_vxl,
                None
                )
        else:
            _backend.grid_encode_backward(
                grad,
                inputs,
                embeddings,
                # embeddings_mask,
                offsets_list,
                resolutions_list,
                grad_embeddings,
                N, num_dim, n_features, n_levels_calc, 0, Rb,
                dy_dx,
                grad_inputs,
                binary_vxl,
                min_level_id
                )

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None, None
grid_encode = _grid_encode.apply
class GridEncoder(nn.Module):
    def __init__(self,
                 num_dim=3,
                 n_features=2,
                 resolutions_list=(16, 23, 32, 46, 64, 92, 128, 184, 256, 368, 512, 736),
                 log2_hashmap_size=19,
                 ste_binary = True,
                 ste_multistep = False,
                 add_noise = False,
                 Q = 1
                 ):
        super().__init__()

        resolutions_list = torch.tensor(resolutions_list).to(torch.int)
        n_levels = resolutions_list.numel()

        self.num_dim = num_dim # coord dims, 2 or 3
        self.n_levels = n_levels # num levels, each level multiply resolution by 2
        self.n_features = n_features # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size
        self.output_dim = n_levels * n_features
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q

        # allocate parameters
        offsets_list = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(n_levels):
            resolution = resolutions_list[i].item()
            params_in_level = min(self.max_params, resolution ** num_dim)  # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible
            offsets_list.append(offset)
            offset += params_in_level
        offsets_list.append(offset)
        offsets_list = torch.from_numpy(np.array(offsets_list, dtype=np.int32))
        self.register_buffer('offsets_list', offsets_list)
        self.register_buffer('resolutions_list', resolutions_list)

        self.n_params = offsets_list[-1] * n_features

        # parameters
        self.params = nn.Parameter(torch.empty(offset, n_features))

        self.reset_parameters()

        self.n_output_dims = n_levels * n_features

    def reset_parameters(self):
        std = 1e-4
        self.params.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: num_dim={self.num_dim} n_levels={self.n_levels} n_features={self.n_features} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.n_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.params.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"

    def forward(self, inputs, min_level_id=None, max_level_id=None, test_phase=False, outspace_params=None, binary_vxl=None, PV=0):
        # inputs: [..., num_dim], normalized real world positions in [0, 1]
        # return: [..., n_levels * n_features]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.num_dim)

        if outspace_params is not None:
            params = nn.Parameter(outspace_params)
        else:
            params = self.params

        if self.ste_binary:
            embeddings = STE_binary.apply(params)
            # embeddings = params
        elif (self.add_noise and not test_phase):
            embeddings = params + (torch.rand_like(params) - 0.5) * (1 / self.Q)
        elif (self.ste_multistep) or (self.add_noise and test_phase):
            embeddings = STE_multistep.apply(params, self.Q)
        else:
            embeddings = params
        # embeddings = embeddings * 0  # for ablation

        min_level_id = 0 if min_level_id is None else max(min_level_id, 0)
        max_level_id = self.n_levels if max_level_id is None else min(max_level_id, self.n_levels)
        n_levels_calc = max_level_id - min_level_id

        outputs = grid_encode(inputs, embeddings, self.offsets_list, self.resolutions_list, inputs.requires_grad, min_level_id, n_levels_calc, binary_vxl, PV)
        outputs = outputs.view(prefix_shape + [n_levels_calc * self.n_features])

        return outputs
