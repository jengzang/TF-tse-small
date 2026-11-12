import torch, torch.nn as nn, torch.nn.functional as F

class GSP(nn.Module):
    # GlobalStatsPool
    def __init__(self,in_planes,acoustic_dim):
        super(GSP, self).__init__()
        self.out_dim = in_planes*8 * 2

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        out = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return out

class GAP (nn.Module):
    # GlobalAvgPool
    def __init__(self,in_planes,acoustic_dim):
        super(GAP, self).__init__()
        self.out_dim = in_planes*8

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        return x.mean(dim=2)

class ASP(nn.Module):
    # Attentive statistics pooling
    def __init__(self,in_planes,acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim/8)
        self.out_dim = in_planes*8 * outmap_size * 2
        
        self.attention = nn.Sequential(
                        nn.Conv1d(in_planes*8 * outmap_size, 128, kernel_size=1),
                        nn.ReLU(),
                        nn.BatchNorm1d(128),
                        nn.Conv1d(128, in_planes*8 * outmap_size, kernel_size=1),
                        nn.Softmax(dim=2),
                        )
        
    def forward(self, x):
        x = x.reshape(x.size()[0],-1,x.size()[-1])
        w = self.attention(x)
        
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
        x = torch.cat((mu,sg),1)
        
        x = x.view(x.size()[0], -1)
        return x
    
class TSP(nn.Module):
    # TemporalStatsPool
    def __init__(self,in_planes,acoustic_dim):
        super(TSP, self).__init__()
        outmap_size = int(acoustic_dim/8)
        self.out_dim = in_planes * 8 * outmap_size * 2
        
    def forward(self, x):
        out_mean = x.mean(dim=-1)
        out_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        out_mean = out_mean.flatten(start_dim=1)
        out_std = out_std.flatten(start_dim=1)
        out = torch.cat((out_mean, out_std), 1)
        return out  
    
class TSDP(nn.Module):
    # TemporalStdPool
    def __init__(self,in_planes,acoustic_dim):
        super(TSDP, self).__init__()
        outmap_size = int(acoustic_dim/8)
        self.out_dim = in_planes*8 * outmap_size 

    def forward(self, x):
        out = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        out = out.flatten(start_dim=1)
        return out 


class MHASTP(torch.nn.Module):
    """ Multi head attentive statistics pooling
    Reference:
        Self Multi-Head Attention for Speaker Recognition
        https://arxiv.org/pdf/1906.09890.pdf
    """

    def __init__(self,
                 in_dim,
                 layer_num=2,
                 head_num=2,
                 d_s=1,
                 bottleneck_dim=64,
                 **kwargs):
        super(MHASTP, self).__init__()
        assert (in_dim % head_num
                ) == 0  # make sure that head num can be divided by input_dim
        self.in_dim = in_dim
        self.head_num = head_num
        d_model = int(in_dim / head_num)
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s
        heads_att_trans = []
        for i in range(self.head_num):
            att_trans = nn.Sequential()
            for i in range(layer_num - 1):
                att_trans.add_module(
                    'att_' + str(i),
                    nn.Conv1d(channel_dims[i], channel_dims[i + 1], 1, 1))
                att_trans.add_module('tanh' + str(i), nn.Tanh())
            att_trans.add_module(
                'att_' + str(layer_num - 1),
                nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num],
                          1, 1))
            heads_att_trans.append(att_trans)
        self.heads_att_trans = nn.ModuleList(heads_att_trans)

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        bs, f_dim, t_dim = input.shape
        chunks = torch.chunk(input, self.head_num, 1)
        # split
        chunks_out = []
        # for i in range(self.head_num):
        #     att_score = self.heads_att_trans[i](chunks[i])
        for i, layer in enumerate(self.heads_att_trans):
            att_score = layer(chunks[i])
            alpha = F.softmax(att_score, dim=-1)
            mean = torch.sum(alpha * chunks[i], dim=2)
            var = torch.sum(alpha * chunks[i]**2, dim=2) - mean**2
            std = torch.sqrt(var.clamp(min=1e-7))
            chunks_out.append(torch.cat((mean, std), dim=1))
        out = torch.cat(chunks_out, dim=1)
        return out   

class MQMHASTP(torch.nn.Module):
    def __init__(self,
                 in_planes,
                 acoustic_dim,
                 layer_num=2,
                 query_num=2,
                 head_num=8,
                 d_s=2,
                 bottleneck_dim=128,
                 **kwargs):
        super(MQMHASTP, self).__init__()
        outmap_size = int(acoustic_dim/8)
        self.query_num = query_num
        self.in_dim = in_planes*8 * outmap_size
        self.out_dim = self.in_dim * 2 * self.query_num
        self.n_query = nn.ModuleList([
            MHASTP(self.in_dim,
                   layer_num=layer_num,
                   head_num=head_num,
                   d_s=d_s,
                   bottleneck_dim=bottleneck_dim) for i in range(query_num)
        ])

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        res = []
        for i, layer in enumerate(self.n_query):
            res.append(layer(input))
        out = torch.cat(res, dim=-1)
        return out





