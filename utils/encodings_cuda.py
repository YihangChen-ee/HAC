import torch
import numpy as np

import arithmetic

chunk_size_cuda = 10000

class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q, input_mean=None):
        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q
        return Q_q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

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


def encoder_factorized_chunk(x, lower_func, Q:float = 1, file_name='tmp.b', chunk_size=1000_0000):
    # should be with 2 dimensions
    # lower_func: xxx._logits_cumulative
    assert file_name.endswith('.b')
    assert len(x.shape) == 2
    N = x.shape[0]
    chunks = int(np.ceil(N / chunk_size))
    bit_len_list = []
    for c in range(chunks):
        bit_len = encoder_factorized(
            x=x[c * chunk_size:c * chunk_size + chunk_size],
            lower_func=lower_func,
            Q=Q,
            file_name=file_name.replace('.b', f'_{str(c)}.b'),
        )
        bit_len_list.append(bit_len)
    return sum(bit_len_list)


def encoder_factorized(x, lower_func, Q:float = 1, file_name='tmp.b'):
    '''
    The reason why int(max_value.item()) + 1 or int(max_value.item()) + 1 + 1:
    first 1: range does not include the last value, so +1
    second 1: if directly calculate, we need to use samples - 0.5, in order to include the whole value space,
              the max bound value after -0.5 should be max_value+0.5.

    Here we do not add the second 1, because we use pmf to calculate cdf, instead of directly calculate cdf

    example in here ("`" means sample-0.5 places, "|" means sample places):
                 `  `  `  `                          `  `  `  `                           `  `  `  `  `
    lkl_lower      |  |  |  |       lkl_upper      |  |  |  |         ->    cdf_lower      |  |  |  |

    example in other place ("`" means sample-0.5 places, "|" means sample places):
                  `  `  `  `  `
    cdf_lower      |  |  |  |

    '''
    # should be with 2 dimensions
    # lower_func: xxx._logits_cumulative
    assert file_name.endswith('.b')
    assert len(x.shape) == 2
    x_int_round = torch.round(x / Q)  # [100]
    max_value = x_int_round.max()
    min_value = x_int_round.min()
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1)).to(
        torch.float).to(x.device)  # from min_value to max_value+1. shape = [max_value+1 - min_value]
    samples = samples.unsqueeze(0).unsqueeze(0).repeat(x.shape[-1], 1, 1)  # [256, 1, max_value+1 - min_value]
    # lower_func: [C, 1, N]
    lower = lower_func((samples - 0.5) * Q, stop_gradient=False)  # [256, 1, max_value+1 - min_value]
    upper = lower_func((samples + 0.5) * Q, stop_gradient=False)  # [256, 1, max_value+1 - min_value]
    sign = -torch.sign(torch.add(lower, upper))
    sign = sign.detach()
    pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))  # [256, 1, max_value+1 - min_value]
    cdf = torch.cumsum(pmf, dim=-1)
    lower = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)  # [256, 1, max_value+1+1 - min_value]
    lower = lower.permute(1, 0, 2).contiguous().repeat(x.shape[0], 1, 1)  # [100, 256, max_value+1+1 - min_value]
    x_int_round_idx = (x_int_round - min_value).to(torch.int16)
    x_int_round_idx = x_int_round_idx.view(-1)  # [100*256]
    lower = lower.view(x_int_round_idx.shape[0], -1)  # [100*256, max_value+1+1 - min_value]
    lower = torch.clamp(lower, min=0.0, max=1.0)
    assert (x_int_round_idx.to(torch.int32) == x_int_round.view(-1) - min_value).all()

    (byte_stream_torch, cnt_torch) = arithmetic.arithmetic_encode(
        x_int_round_idx,
        lower,
        chunk_size_cuda,
        int(lower.shape[0]),
        int(lower.shape[1])
    )
    cnt_bytes = cnt_torch.cpu().numpy().tobytes()
    byte_stream_bytes = byte_stream_torch.cpu().numpy().tobytes()
    len_cnt_bytes = len(cnt_bytes)
    with open(file_name, 'wb') as fout:
        fout.write(min_value.to(torch.float32).cpu().numpy().tobytes())
        fout.write(max_value.to(torch.float32).cpu().numpy().tobytes())
        fout.write(np.array([len_cnt_bytes]).astype(np.int32).tobytes())
        fout.write(cnt_bytes)
        fout.write(byte_stream_bytes)
    bit_len = (len(byte_stream_bytes) + len(cnt_bytes)) * 8 + 32 * 3
    return bit_len

def decoder_factorized_chunk(lower_func, Q, N_len, dim, file_name='tmp.b', device='cuda', chunk_size=1000_0000):
    assert file_name.endswith('.b')
    chunks = int(np.ceil(N_len / chunk_size))
    x_c_list = []
    for c in range(chunks):
        x_c = decoder_factorized(
            lower_func=lower_func,
            Q=Q,
            N_len=min(chunk_size, N_len-c*chunk_size),
            dim=dim,
            file_name=file_name.replace('.b', f'_{str(c)}.b'),
            device=device,
        )
        x_c_list.append(x_c)
    x_c_list = torch.cat(x_c_list, dim=0)
    return x_c_list


def decoder_factorized(lower_func, Q, N_len, dim, file_name='tmp.b', device='cuda'):
    assert file_name.endswith('.b')

    with open(file_name, 'rb') as fin:
        min_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device="cuda")
        max_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device="cuda")
        len_cnt_bytes = np.frombuffer(fin.read(4), dtype=np.int32)[0]
        cnt_torch = torch.tensor(np.frombuffer(fin.read(len_cnt_bytes), dtype=np.int32).copy(), device="cuda")
        byte_stream_torch = torch.tensor(np.frombuffer(fin.read(), dtype=np.uint8).copy(), device="cuda")

    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1)).to(
        torch.float).to(device)  # from min_value to max_value+1. shape = [max_value+1 - min_value]
    samples = samples.unsqueeze(0).unsqueeze(0).repeat(dim, 1, 1)  # [256, 1, max_value+1 - min_value]

    # lower_func: [C, 1, N]
    lower = lower_func((samples - 0.5) * Q, stop_gradient=False)  # [256, 1, max_value+1 - min_value]
    upper = lower_func((samples + 0.5) * Q, stop_gradient=False)  # [256, 1, max_value+1 - min_value]
    sign = -torch.sign(torch.add(lower, upper))
    sign = sign.detach()
    pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))  # [256, 1, max_value+1 - min_value]
    cdf = torch.cumsum(pmf, dim=-1)
    lower = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)  # [256, 1, max_value+1+1 - min_value]
    lower = lower.permute(1, 0, 2).contiguous().repeat(N_len, 1, 1)  # [100, 256, max_value+1+1 - min_value]
    lower = lower.view(N_len*dim, -1)  # [100*256, max_value+1+1 - min_value]
    lower = torch.clamp(lower, min=0.0, max=1.0)

    sym_out = arithmetic.arithmetic_decode(
        lower,
        byte_stream_torch,
        cnt_torch,
        chunk_size_cuda,
        int(lower.shape[0]),
        int(lower.shape[1])
    ).to(device).to(torch.float32)
    x = sym_out + min_value
    x = x * Q
    x = x.reshape(N_len, dim)
    return x


def encoder_gaussian_mixed_chunk(x, mean_list, scale_list, prob_list, Q, file_name='tmp.b', chunk_size=1000_0000):
    assert file_name.endswith('.b')
    assert len(x.shape) == 1
    x_view = x.view(-1)
    mean_list_view = [mean.view(-1) for mean in mean_list]
    scale_list_view = [scale.view(-1) for scale in scale_list]
    prob_list_view = [prob.view(-1) for prob in prob_list]
    assert x_view.shape[0]==mean_list_view[0].shape[0]==scale_list_view[0].shape[0]==prob_list_view[0].shape[0]
    N = x_view.shape[0]
    chunks = int(np.ceil(N/chunk_size))
    Is_Q_tensor = isinstance(Q, torch.Tensor)
    if Is_Q_tensor: Q_view = Q.view(-1)
    bit_len_list = []
    for c in range(chunks):
        bit_len = encoder_gaussian_mixed(
            x=x_view[c*chunk_size:c*chunk_size + chunk_size],
            mean_list=[mean[c*chunk_size:c*chunk_size + chunk_size] for mean in mean_list_view],
            scale_list=[scale[c*chunk_size:c*chunk_size + chunk_size] for scale in scale_list_view],
            prob_list=[prob[c*chunk_size:c*chunk_size + chunk_size] for prob in prob_list_view],
            Q=Q_view[c*chunk_size:c*chunk_size + chunk_size] if Is_Q_tensor else Q,
            file_name=file_name.replace('.b', f'_{str(c)}.b'),
        )
        bit_len_list.append(bit_len)
    return sum(bit_len_list)


def encoder_gaussian_mixed(x, mean_list, scale_list, prob_list, Q, file_name='tmp.b'):
    # should be with single dimension
    assert file_name.endswith('.b')
    assert len(x.shape) == 1
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=x.dtype, device=x.device).repeat(x.shape[0])
    assert x.shape == mean_list[0].shape == scale_list[0].shape == prob_list[0].shape == Q.shape, f'{x.shape}, {mean_list[0].shape}, {scale_list[0].shape}, {prob_list[0].shape}, {Q.shape}'
    x_int_round = torch.round(x / Q)  # [100]
    max_value = x_int_round.max()
    min_value = x_int_round.min()
    lower_all = int(0)
    for (mean, scale, prob) in zip(mean_list, scale_list, prob_list):
        lower = arithmetic.calculate_cdf(
            mean,
            scale,
            Q,
            min_value,
            max_value
        ) * prob.unsqueeze(-1)
        if isinstance(lower_all, int):
            lower_all = lower
        else:
            lower_all += lower
    lower = torch.clamp(lower_all, min=0.0, max=1.0)
    del mean
    del scale
    del prob

    x_int_round_idx = (x_int_round - min_value).to(torch.int16)
    (byte_stream_torch, cnt_torch) = arithmetic.arithmetic_encode(
        x_int_round_idx,
        lower,
        chunk_size_cuda,
        int(lower.shape[0]),
        int(lower.shape[1])
    )
    cnt_bytes = cnt_torch.cpu().numpy().tobytes()
    byte_stream_bytes = byte_stream_torch.cpu().numpy().tobytes()
    len_cnt_bytes = len(cnt_bytes)
    with open(file_name, 'wb') as fout:
        fout.write(min_value.to(torch.float32).cpu().numpy().tobytes())
        fout.write(max_value.to(torch.float32).cpu().numpy().tobytes())
        fout.write(np.array([len_cnt_bytes]).astype(np.int32).tobytes())
        fout.write(cnt_bytes)
        fout.write(byte_stream_bytes)
    bit_len = (len(byte_stream_bytes) + len(cnt_bytes))*8 + 32 * 3
    return bit_len


def decoder_gaussian_mixed_chunk(mean_list, scale_list, prob_list, Q, file_name='tmp.b', chunk_size=1000_0000):
    assert file_name.endswith('.b')
    mean_list_view = [mean.view(-1) for mean in mean_list]
    scale_list_view = [scale.view(-1) for scale in scale_list]
    prob_list_view = [prob.view(-1) for prob in prob_list]
    N = mean_list_view[0].shape[0]
    chunks = int(np.ceil(N/chunk_size))
    Is_Q_tensor = isinstance(Q, torch.Tensor)
    if Is_Q_tensor: Q_view = Q.view(-1)
    x_c_list = []
    for c in range(chunks):
        x_c = decoder_gaussian_mixed(
            mean_list=[mean[c*chunk_size:c*chunk_size + chunk_size] for mean in mean_list_view],
            scale_list=[scale[c*chunk_size:c*chunk_size + chunk_size] for scale in scale_list_view],
            prob_list=[prob[c*chunk_size:c*chunk_size + chunk_size] for prob in prob_list_view],
            Q=Q_view[c*chunk_size:c*chunk_size + chunk_size] if Is_Q_tensor else Q,
            file_name=file_name.replace('.b', f'_{str(c)}.b'),
        )
        x_c_list.append(x_c)
    x_c_list = torch.cat(x_c_list, dim=0).type_as(mean_list[0])
    return x_c_list


def decoder_gaussian_mixed(mean_list, scale_list, prob_list, Q, file_name='tmp.b'):
    assert file_name.endswith('.b')
    m0 = mean_list[0]
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=m0.dtype, device=m0.device).repeat(m0.shape[0])
    assert mean_list[0].shape == scale_list[0].shape == prob_list[0].shape == Q.shape

    with open(file_name, 'rb') as fin:
        min_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device="cuda")
        max_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device="cuda")
        len_cnt_bytes = np.frombuffer(fin.read(4), dtype=np.int32)[0]
        cnt_torch = torch.tensor(np.frombuffer(fin.read(len_cnt_bytes), dtype=np.int32).copy(), device="cuda")
        byte_stream_torch = torch.tensor(np.frombuffer(fin.read(), dtype=np.uint8).copy(), device="cuda")

    lower_all = int(0)
    for (mean, scale, prob) in zip(mean_list, scale_list, prob_list):
        lower = arithmetic.calculate_cdf(
            mean,
            scale,
            Q,
            min_value,
            max_value
        ) * prob.unsqueeze(-1)
        if isinstance(lower_all, int):
            lower_all = lower
        else:
            lower_all += lower
    lower = torch.clamp(lower_all, min=0.0, max=1.0)

    sym_out = arithmetic.arithmetic_decode(
        lower,
        byte_stream_torch,
        cnt_torch,
        chunk_size_cuda,
        int(lower.shape[0]),
        int(lower.shape[1])
    ).to(mean.device).to(torch.float32)
    x = sym_out + min_value
    x = x * Q
    return x


def encoder_gaussian_chunk(x, mean, scale, Q, file_name='tmp.b', chunk_size=1000_0000):
    assert file_name.endswith('.b')
    assert len(x.shape) == 1
    x_view = x.view(-1)
    mean_view = mean.view(-1)
    scale_view = scale.view(-1)
    N = x_view.shape[0]
    chunks = int(np.ceil(N/chunk_size))
    Is_Q_tensor = isinstance(Q, torch.Tensor)
    if Is_Q_tensor: Q_view = Q.view(-1)
    bit_len_list = []
    for c in range(chunks):
        bit_len = encoder_gaussian(
            x=x_view[c*chunk_size:c*chunk_size + chunk_size],
            mean=mean_view[c*chunk_size:c*chunk_size + chunk_size],
            scale=scale_view[c*chunk_size:c*chunk_size + chunk_size],
            Q=Q_view[c*chunk_size:c*chunk_size + chunk_size] if Is_Q_tensor else Q,
            file_name=file_name.replace('.b', f'_{str(c)}.b'),
        )
        bit_len_list.append(bit_len)
    return sum(bit_len_list)


def encoder_gaussian(x, mean, scale, Q, file_name='tmp.b'):
    # should be single dimension
    assert file_name.endswith('.b')
    assert len(x.shape) == 1
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    x_int_round = torch.round(x / Q)  # [100]
    max_value = x_int_round.max()
    min_value = x_int_round.min()

    lower = arithmetic.calculate_cdf(
        mean,
        scale,
        Q,
        min_value,
        max_value
    )

    x_int_round_idx = (x_int_round - min_value).to(torch.int16)
    (byte_stream_torch, cnt_torch) = arithmetic.arithmetic_encode(
        x_int_round_idx,
        lower,
        chunk_size_cuda,
        int(lower.shape[0]),
        int(lower.shape[1])
    )
    cnt_bytes = cnt_torch.cpu().numpy().tobytes()
    byte_stream_bytes = byte_stream_torch.cpu().numpy().tobytes()
    len_cnt_bytes = len(cnt_bytes)
    with open(file_name, 'wb') as fout:
        fout.write(min_value.to(torch.float32).cpu().numpy().tobytes())
        fout.write(max_value.to(torch.float32).cpu().numpy().tobytes())
        fout.write(np.array([len_cnt_bytes]).astype(np.int32).tobytes())
        fout.write(cnt_bytes)
        fout.write(byte_stream_bytes)
    bit_len = (len(byte_stream_bytes) + len(cnt_bytes))*8 + 32 * 3
    return bit_len

def decoder_gaussian_chunk(mean, scale, Q, file_name='tmp.b', chunk_size=1000_0000):
    assert file_name.endswith('.b')
    mean_view = mean.view(-1)
    scale_view = scale.view(-1)
    N = mean_view.shape[0]
    chunks = int(np.ceil(N/chunk_size))
    Is_Q_tensor = isinstance(Q, torch.Tensor)
    if Is_Q_tensor: Q_view = Q.view(-1)
    x_c_list = []
    for c in range(chunks):
        x_c = decoder_gaussian(
            mean=mean_view[c*chunk_size:c*chunk_size + chunk_size],
            scale=scale_view[c*chunk_size:c*chunk_size + chunk_size],
            Q=Q_view[c*chunk_size:c*chunk_size + chunk_size] if Is_Q_tensor else Q,
            file_name=file_name.replace('.b', f'_{str(c)}.b'),
        )
        x_c_list.append(x_c)
    x_c_list = torch.cat(x_c_list, dim=0).type_as(mean)
    return x_c_list


def decoder_gaussian(mean, scale, Q, file_name='tmp.b'):
    # should be single dimension
    assert file_name.endswith('.b')
    assert len(mean.shape) == 1
    assert mean.shape == scale.shape
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])

    with open(file_name, 'rb') as fin:
        min_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device="cuda")
        max_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device="cuda")
        len_cnt_bytes = np.frombuffer(fin.read(4), dtype=np.int32)[0]
        cnt_torch = torch.tensor(np.frombuffer(fin.read(len_cnt_bytes), dtype=np.int32).copy(), device="cuda")
        byte_stream_torch = torch.tensor(np.frombuffer(fin.read(), dtype=np.uint8).copy(), device="cuda")

    lower = arithmetic.calculate_cdf(
        mean,
        scale,
        Q,
        min_value,
        max_value
    )

    sym_out = arithmetic.arithmetic_decode(
        lower,
        byte_stream_torch,
        cnt_torch,
        chunk_size_cuda,
        int(lower.shape[0]),
        int(lower.shape[1])
    ).to(mean.device).to(torch.float32)
    x = sym_out + min_value
    x = x * Q
    return x


def encoder(x, file_name='tmp.b'):
    # x: 0 or 1
    assert file_name[-2:] == '.b'
    x = x.detach().view(-1)
    p = torch.zeros_like(x).to(torch.float32)
    prob_1 = x.sum() / x.numel()
    p[...] = prob_1
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    sym = torch.floor(x).to(torch.int16)
    (byte_stream_torch, cnt_torch) = arithmetic.arithmetic_encode(
        sym,
        output_cdf,
        chunk_size_cuda,
        int(output_cdf.shape[0]),
        int(output_cdf.shape[1])
    )
    cnt_bytes = cnt_torch.cpu().numpy().tobytes()
    byte_stream_bytes = byte_stream_torch.cpu().numpy().tobytes()
    len_cnt_bytes = len(cnt_bytes)
    with open(file_name, 'wb') as fout:
        fout.write(prob_1.to(torch.float32).cpu().numpy().tobytes())
        fout.write(np.array([len_cnt_bytes]).astype(np.int32).tobytes())
        fout.write(cnt_bytes)
        fout.write(byte_stream_bytes)
    bit_len = (len(byte_stream_bytes) + len(cnt_bytes)) * 8 + 32 * 2
    return bit_len


def decoder(N_len, file_name='tmp.b', device='cuda'):
    assert file_name[-2:] == '.b'

    with open(file_name, 'rb') as fin:
        prob_1 = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy())
        len_cnt_bytes = np.frombuffer(fin.read(4), dtype=np.int32)[0]
        cnt_torch = torch.tensor(np.frombuffer(fin.read(len_cnt_bytes), dtype=np.int32).copy(), device="cuda")
        byte_stream_torch = torch.tensor(np.frombuffer(fin.read(), dtype=np.uint8).copy(), device="cuda")
    p = torch.zeros(size=[N_len], dtype=torch.float32, device="cuda")
    p[...] = prob_1
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    # Read from a file.
    # Decode from bytestream.
    sym_out = arithmetic.arithmetic_decode(
        output_cdf,
        byte_stream_torch,
        cnt_torch,
        chunk_size_cuda,
        int(output_cdf.shape[0]),
        int(output_cdf.shape[1])
    )
    return sym_out

