import torch
import torch.nn as nn
import torch.nn.functional as F
from lvae.decoders.LSTM_decoder import LSTM_decoder
from lvae.encoders.LSTM_encoder import LSTM_encoder

class LVAE(torch.nn.Module):
    def __init__(self,vocab,config):
        super(LVAE,self).__init__()
        # for ss in ('bos', 'eos', 'unk', 'pad'):
        #     setattr(self, ss, getattr(vocab, ss))
        self.d_size = config.ladder_d_size
        self.z_size = config.ladder_z_size
        self.z2z_layer_size = config.ladder_z2z_layer_size
        self.full_z_size = sum(config.ladder_z_size)
        self.z_reverse_size = list(reversed(config.ladder_z_size))
        self.vocab = vocab
        # Word embeddings layer
        self.embedding = nn.Embedding(len(self.vocab.i2c),config.emb_sz,self.vocab.pad)

        # Encoder
        # if config.enc_type == 'lstm':
        #     self.encoder = LSTM_encoder(self.embedding,self.vocab,config)
        #     self.ladder_input_size = config.enc_hidden_size * (1 + int(config.enc_bidirectional))
        # else :
        #     raise ValueError(
        #         "Invalid encoder_type"
        #     )
        self.encoder = nn.Linear(h_size, embedding_size)  # mu


        # Decoder
        # if config.dec_type == 'lstm':
        #     self.decoder = LSTM_decoder(self.vocab,self.embedding,config,self.full_z_size)
        # else:
        #     raise ValueError(
        #         "Invalid decoder_type"
        #     )
        self.decoder_1 = nn.Linear(embedding_size, embedding_size)
        self.decoder_2 = nn.Linear(embedding_size, y_size)
        self.relu = nn.ReLU()

        #ladder
        self.top_down_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

        #get a list contain bottom up layers,which used to generate z_mu_q_d and z_log_var_q_d
        for i in range(len(self.d_size)):
            if i == 0:
                self.bottom_up_layers.append(MLP(in_size=self.ladder_input_size,layer_size=self.d_size[i],out_size=self.z_size[i]))
            else:
                self.bottom_up_layers.append(MLP(in_size=self.d_size[i-1],layer_size=self.d_size[i],out_size=self.z_size[i]))

        #get a list contain top down layers,which used to generate z_mu_p and z_log_var_p
        for i in range(len(self.z_reverse_size)-1):
            self.top_down_layers.append(MLP(in_size=self.z_reverse_size[i],layer_size=self.z2z_layer_size[i],out_size=self.z_reverse_size[i+1]))

    def device(self):
        return next(self.parameters()).device

    def bottom_up(self,input):
        '''
        Do the bottom up step and get z_mu_q_d and z_log_var_q_d

        :param: input of ladder part,size=(batch_size * ladder_input_size)
        :return:two lists: z_mu_q_d ,z_log_var_q_d
        '''
        z_mu_q_d = []
        z_log_var_q_d = []
        for i in range(len(self.d_size)):
            if i == 0:
                nn, mu_q_d, log_var_q_d = self.bottom_up_layers[i](input)
                z_mu_q_d.append(mu_q_d)
                z_log_var_q_d.append(log_var_q_d)
            else:
                nn, mu_q_d, log_var_q_d = self.bottom_up_layers[i](nn)
                z_mu_q_d.append(mu_q_d)
                z_log_var_q_d.append(log_var_q_d)
        return z_mu_q_d, z_log_var_q_d # shape:[[batch_size * z_size[0]],[batch_size * z_size[1]],....,[batch_size * z_size[-1]]]

    def top_down(self, z_mu_q_d,z_log_var_q_d, z_sample=None, mode="train"):
        '''
        Do top down step and get z_mu_p,z_log_var_p,z_mu_q,z_log_var_q, also return samples of each level z

        :param z_mu_q_d: z_mu_q_d from bottom up step
        :param z_log_var_q_d: z_log_var_q_d from bottom up step
        :return five list :z_mu_p,z_log_var_p,z_sample,z_mu_q,z_log_var_q
        
        
        Note:
            * z_sample is not None when evaluating test set reconstruction
        
        '''
        #initialize required list
        z_mu_p = []
        z_log_var_p = []
        z_mu_q = []
        z_log_var_q = []
        z_mu_p.append(torch.zeros(z_mu_q_d[-1].size()).to(self.device())) #[[batch_size * z_size[-1]]]
        z_log_var_p.append(torch.zeros(z_log_var_q_d[-1].size()).to(self.device())) #[[batch_size * z_size[-1]]]
        z_mu_q.append(z_mu_q_d[-1])#[[batch_size * z_size[-1]]]
        z_log_var_q.append(z_log_var_q_d[-1])#[[batch_size * z_size[-1]]]
        
        if z_sample is None and mode == "train":
            z_sample = []
            z_sample.append(self.sample_z(z_mu_q[0], z_log_var_q[0]))  # [[batch_size * z_size[-1]]]
        else:
            assert z_sample is not None

        for i in range(len(self.z_reverse_size) - 1):
            _, mu_p, log_var_p = self.top_down_layers[i](z_sample[i])
            z_mu_p.append(mu_p)
            z_log_var_p.append(log_var_p)

            # combine z_mu_q_d, z_log_var_q_d, z_mu_p, z_log_var_p to generate z_mu_q and z_log_var_q
            mu_q,log_var_q = self.Gaussian_update(z_mu_q_d[-i-2], z_log_var_q_d[-i-2], z_mu_p[i+1], z_log_var_p[i+1])
            z_mu_q.append(mu_q)
            z_log_var_q.append(log_var_q)

            #sample z from z_mu_q,z_log_var_q
            z_sample.append(self.sample_z(mu_q,log_var_q))
        #the shape of z_mu_p,z_log_var_p z_mu_q,z_log_var_q and z_sample after loop:[[batch_size * z_size[-1]],[batch_size * z_size[-2]],...[batch_size * z_size[0]]]
        return list(reversed(z_mu_p)), list(reversed(z_log_var_p)), list(reversed(z_mu_q)),list(reversed(z_log_var_q)),list(reversed(z_sample))#reverse lists of z_mu_p,z_log_var_p,z_mu_q,z_log_var_q and z_sample

    def sample_z(self, mu, log_var):
        '''
        Sampling z ~ p(z)= N(mu,var)
        :return: sample of z
        '''
        stddev = torch.exp(log_var) ** 0.5
        out = mu + stddev * torch.randn(mu.size()).to(self.device())
        return out

    def Gaussian_update(self, mu_q_d, log_var_q_d, mu_p, log_var_p):
        '''
        Combine mu_q_d,var_q_d and mu_p,var_p to generate mu_q,var_q
        :return two tensor: mu_q,var_q
        '''
        var = 1/(torch.exp(-log_var_q_d) + torch.exp(-log_var_p))
        mu = (mu_q_d*torch.exp(-log_var_q_d) + mu_p*torch.exp(-log_var_p)) / (torch.exp(-log_var_q_d) + torch.exp(-log_var_p))
        
        return mu, torch.log(var)

    def KL_loss(self,q_mu,q_log_var,p_mu,p_log_var):
        q_var, p_var = torch.exp(q_log_var),torch.exp(p_log_var)
        kl = 0.5*(p_log_var - q_log_var + q_var/p_var + torch.pow(torch.add(q_mu,-p_mu),2)/p_var -1)
        return kl.sum(1).mean()

    def forward_latent(self,input):
        # initialize required variable
        kl_loss = 0

        #do bottom up step and get z_mu_q_d, z_log_var_q_d
        z_mu_q_d, z_log_var_q_d = self.bottom_up(input)# [[batch_size * z_size[0]],[batch_size * z_size[1]],...,[batch_size * z_size[-1]]]

        #do top down step and get z_mu_p, z_log_var_p,z_sample
        z_mu_p, z_log_var_p, z_mu_q,z_log_var_q,z_sample= self.top_down(z_mu_q_d,z_log_var_q_d)# [[batch_size * z_size[0]],[batch_size * z_size[1]],...,[batch_size * z_size[-1]]]

        # concatenate z_sample
        z_out = z_sample[0]
        for i in range(len(self.z_size)-1):
            z_out = torch.cat((z_out,z_sample[i+1]),1) # batch_size * full_z_size

        #calculate KL_loss
        for i in range(len(self.z_size)):
            kl_loss = kl_loss + self.KL_loss(z_mu_q[i], z_log_var_q[i], z_mu_p[i], z_log_var_p[i])
        return z_out,kl_loss

    def forward(self, batch):
        _,h = self.encoder(batch)
        z,KL_loss = self.forward_latent(h)
        #  b

        recon_loss = self.decoder(batch,z)
        return KL_loss,recon_loss

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocab.ids2string(ids, rem_bos=True, rem_eos=True)
        return string

    def gen_top_down(self,z_sample, z_mu_p, z_log_var_p):
        '''
        Top down step during generation only

        :param z_sample: only have the samples of top z
        :param z_mu_p: only have the z_mu_p of top z (mu = 0)
        :param z_log_var_p: only have the z_log_var_p of top z (var = 1)
        :return three list :z_mu_p,z_log_var_p,z_sample
        '''
        for i in range(len(self.z_reverse_size) - 1):
            _, mu_p, log_var_p = self.top_down_layers[i](z_sample[i])
            z_sample.append(self.sample_z(mu_p, log_var_p))
            z_mu_p.append(mu_p)
            z_log_var_p.append(log_var_p)
        #the shape of z_mu_p,z_log_var_p and z_sample after loop:[[batch_size * z_size[-1]],[batch_size * z_size[-2]],...[batch_size * z_size[0]]]
        return list(reversed(z_mu_p)), list(reversed(z_log_var_p)), list(reversed(z_sample))#reverse lists of z_mu_p,z_log_var_p and z_sample
    
    def get_z_control(self, sample_layer, num_of_sample, layer_num):
        '''
        sample n times at special z layer
        difference from `sample_z` func: return a complete all-level sample with
    
        :param z_mu_p: z_mu_p of each level z from top-down step
        :param z_log_var_p:z_log_var_p of each level z from top-down step
        :param z_sample:z_sample of each level z from top-down step
        :param sample_layer: the index of z layer needing sample multiple times
        :param num_of_sample: sample times of special z layer
        :param layer_num: number of z layers
    
        :return: a list contain sampled z at each level
        '''
        # get one sample from top down step
        z_mu_p = []
        z_log_var_p = []
        z_sample = []
        z_mu_p.append(torch.zeros(1, self.z_size[-1]).to(self.device()))
        z_log_var_p.append(torch.zeros(1, self.z_size[-1]).to(self.device()))
        z_sample.append(self.sample_z(z_mu_p[0], z_log_var_p[0]))
        z_mu_p,z_log_var_p,z_sample = self.gen_top_down(z_sample,z_mu_p,z_log_var_p)
        
        # repeat z layers(>sample_layer) n times
        for i in range(layer_num - sample_layer - 1):
            z_sample[-i - 1] = z_sample[-i - 1].repeat(num_of_sample, 1)
    
        # sample special z layer n times
        for i in range(num_of_sample):
            z = self.sample_z(z_mu_p[-layer_num + sample_layer], z_log_var_p[-layer_num + sample_layer])
            if i == 0:
                z_sample[-layer_num + sample_layer] = z
            else:
                z_sample[-layer_num + sample_layer] = torch.cat((z_sample[-layer_num + sample_layer], z), dim=0)
    
        # calculate mu of z layers(<sample_layer),which used as sampled z
        for i in range(sample_layer):
            for j in range(num_of_sample):
                if j == 0:
                    _, mu, _ = self.top_down_layers[layer_num - sample_layer + i - 1](
                        z_sample[-layer_num + sample_layer - i][j])
                    mu = mu.unsqueeze(0)
                    z_sample[-layer_num + sample_layer - i - 1] = mu
                else:
                    _, mu, _ = self.top_down_layers[layer_num - sample_layer + i - 1](
                        z_sample[-layer_num + sample_layer - i][j])
                    mu = mu.unsqueeze(0)
                    z_sample[-layer_num + sample_layer - i - 1] = torch.cat(
                        (z_sample[-layer_num + sample_layer - i - 1], mu), dim=0)
        return z_sample
    
    def get_z_prior(self, n_enc_zs):
        """Get multiple samples at each z layer
        e.g. Sample m times at top z layer. 
             Sample n times for each z_top at next highest layer and get m*n unique samples  
        """
        
        # get sampling times at each level z
        zs = n_enc_zs
        if len(set(zs))==1:
            z_mu_p = []
            z_log_var_p = []
            z_sample = []
            z_mu_p.append(torch.zeros(zs[0], self.z_size[-1]).to(self.device()))
            z_log_var_p.append(torch.zeros(zs[0], self.z_size[-1]).to(self.device()))
            z_sample.append(self.sample_z(z_mu_p[0], z_log_var_p[0]))
            _,_,z_sample_re = self.gen_top_down(z_sample,z_mu_p,z_log_var_p)
        else:
            repeat_times = [int(zs[i]/zs[i+1]) for i in range(len(zs)-1)] + [zs[-1]]
            repeat_times = list(reversed(repeat_times)) # top z -> bottom z
            
            z_size = self.z_size
            z_sample = []
            z_sample_re =[]
            mu = torch.zeros((repeat_times[0],z_size[-1])).to(self.device())
            log_var = torch.zeros((repeat_times[0],z_size[-1])).to(self.device())
            z_sample.append(self.sample_z(mu,log_var))
            z_sample_re.append(z_sample[0].repeat_interleave(int(zs[0]/zs[-1]),dim=0))
            
            for i in range(len(z_size)-1):
                _,mu,log_var = self.top_down_layers[i](z_sample[i])
                for j in range(repeat_times[-i-2]):
                    if j == 0 :
                        z = self.sample_z(mu,log_var)
                        z_sample.append(z)
                    else:
                        z = self.sample_z(mu, log_var)
                        z_sample[i+1] = torch.cat((z_sample[i+1],z),dim=1)
                        
                z_sample[i+1] = z_sample[i+1].view(-1,z_size[-2-i])
                z_sample_re.append(z_sample[i+1].repeat_interleave(int(zs[0]/zs[-i-2]),dim=0)) # if repeat 1 remains unchange
            #z_sample = list(reversed(z_sample))
            z_sample_re = list(reversed(z_sample_re))#list contain each level z,unconcatenated
        
        return z_sample_re
        
    
    def sample(self,n_enc_zs,max_len=100,temp=1.0,
               z_in=None,concated=False,deterministic=False,
               n_dec_times=1, sample_type="prior", sample_layer=None):
        '''
        Get z and decode into x. Dafault: get z from top ladder layer.
        Generating n_batch*n_dec_times samples.

        :param n_batch: number of sentences to generate (deprecated)
        :param max_len: max len of samples
        :param n_enc_zs: number of unique samples in each z layer
        :param temp: temperature of softmax
        :param z_in: could be 
                        [(batch_size, z_size[0]),(batch_size, z_size[1]),...,(batch_size, z_size[-1])] , list of tensor of latent z. Default: None
                     or
                        tensor(batch_size, sum(z_size)) concatenated each z layer
        :param concated: whether z_in is concatenated
        :param deterministic: do random sampling or use argmax
        :param n_dec_times: decode n sequences from each z
        :param sample_type: way to get complete z samples. Choose from "prior" and "control_z"
        :param sample_layer: index of z layer to sample from when do "control_z" sampling
        
        :return: list of tensors of strings, samples sequence x
        '''
        with torch.no_grad():
            n_batch = n_enc_zs[0]
            
            # get samples of  z
            if z_in is not None:
                z_sample = z_in
            else:
                if sample_type == "prior":
                    z_sample = self.get_z_prior(n_enc_zs)
                elif sample_type == "control_z":
                    assert sample_layer is not None, "Should specify layer index to sample from!"
                    assert isinstance(sample_layer, int), "Expect int type with sample_layer!"
                    z_sample = self.get_z_control(sample_layer,n_batch,len(self.z_size))

            # concatenate z_sample
            if not concated or z_in is None:
                cat_z = z_sample[0]
                for i in range(len(self.z_size) - 1):
                    cat_z = torch.cat((cat_z, z_sample[i + 1]), 1)
                z = cat_z.unsqueeze(1) # n_batch * 1 * full_z_size
            else:
                cat_z = z_sample[:]
                z = z_sample.unsqueeze(1)
            
            # repeat to decode multiple x for each z
            if n_dec_times > 1:
                cat_z = cat_z.repeat_interleave(n_dec_times, dim=0)
                z = z.repeat_interleave(n_dec_times, dim=0)
                n_samples = n_batch * n_dec_times
            else:
                n_samples = n_batch

            # inital values
            h = self.decoder.map_z2hc(cat_z) # n_samples * dec_hid_sz *2
            h_0, c_0 = h[:,:self.decoder.d_d_h], h[:,self.decoder.d_d_h:] # n_samples * dec_hid_sz
            h_0 = h_0.unsqueeze(0).repeat(self.decoder.lstm.num_layers, 1, 1)  # dec_n_layer * n_samples * dec_hid_sz
            c_0 = c_0.unsqueeze(0).repeat(self.decoder.lstm.num_layers, 1, 1)  # dec_n_layer * n_samples * dec_hid_sz
            w = torch.tensor(self.bos).repeat(n_samples).to(self.device()) # n_samples
            x = torch.tensor([self.pad]).repeat(n_samples,max_len).to(self.device()) # n_samples * max_len

            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len]).to(self.device()).repeat(n_samples) # a tensor record the length of each molecule, size=n_batch
            eos_mask = torch.zeros(n_samples, dtype=torch.bool).to(self.device()) # a tensor indicate the molecular generation process is over or not,size=n_batch

            # generating cycle
            for i in range(1,max_len):
                x_emb = self.embedding(w).unsqueeze(1) # n_samples * 1 * embed_size
                x_input = torch.cat([x_emb, z], dim=-1) # n_samples * 1 * (embed_size + full_z_size)
                output, (h_0,c_0) = self.decoder.lstm(x_input, (h_0,c_0)) # output size : n_samples * 1 * dec_hid_sz
                y = self.decoder.decoder_fc(output.squeeze(1))
                y = F.softmax(y / temp, dim=-1) # n_samples * n_vocab

                if deterministic:
                    w = torch.max(y,1)[1]
                else:
                    w = torch.multinomial(y, 1)[:, 0] # input of next generate step, size=n_samples
                x[~eos_mask, i] = w[~eos_mask] # add generated atom to molecule
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1 # update end_pads
                eos_mask = eos_mask | i_eos_mask #update eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])
            return [self.tensor2string(i_x) for i_x in new_x]


class MLP(torch.nn.Module):
    def __init__(self,in_size,layer_size,out_size):
        super(MLP,self).__init__()
        self.layer1 = nn.Linear(in_size,layer_size, bias=False)
        self.layer2 = nn.Linear(layer_size,layer_size, bias=False)
        self.mu = nn.Linear(layer_size,out_size)
        self.logvar = nn.Linear(layer_size,out_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)

    def forward(self,input):
        layer1 = self.layer1(input)
        layer1 = self.bn1(layer1)
        layer1 = F.leaky_relu(layer1)
        layer2 = self.layer2(layer1)
        layer2 = self.bn2(layer2)
        layer2 = F.leaky_relu(layer2)
        
        mu = self.mu(layer2)
        logvar = self.logvar(layer2)
        return layer2,mu,logvar


import argparse


def get_parser(parser=None):
    if parser is None:  # ? save for subparser
        parser = argparse.ArgumentParser()

    ########## experiment
    # expr_args = parser.add_argument_group("experiment")
    # add_expr_parser(expr_args)

    ########## model
    model_args = parser.add_argument_group('Model')
    model_args.add_argument("--enc_type",
                            type=str, default="lstm",
                            help="Encoder type")
    model_args.add_argument("--dec_type",
                            type=str, default="lstm",
                            help="Decoder type")

    ## encoder
    model_args.add_argument("--emb_sz",
                            type=int, default=128,
                            help="Embedding size")
    model_args.add_argument("--enc_hidden_size",
                            type=int, default=256,
                            help="Encoder hidden vector size")
    model_args.add_argument("--enc_num_layers",
                            type=int, default=1,
                            help="Encoder number of lstm layers")
    model_args.add_argument("--enc_sorted_seq",
                            type=bool, default=True,
                            help="If the input of encoder is not sorted,set Flase")
    model_args.add_argument("--enc_bidirectional",
                            type=bool, default=True,
                            help="If True, becomes a bidirectional LSTM_encoder")

    ## decoder
    model_args.add_argument("--dec_hid_sz",
                            type=int, default=256,
                            help="Decoder hidden vector size")
    model_args.add_argument("--dec_n_layer",
                            type=int, default=1,
                            help="Decoder number of lstm layers")

    ## ladder latent code
    model_args.add_argument("--ladder_d_size",
                            type=int, default=[128, 64, 32],
                            nargs="+",
                            help="The dimension of each layer in deterministic upward")
    model_args.add_argument("--ladder_z_size",
                            type=int, default=[16, 8, 4],
                            nargs="+",
                            help="The dimension of each level latent z")
    model_args.add_argument("--ladder_z2z_layer_size",
                            type=int, default=[8, 16],
                            nargs="+",
                            help="The z2z layer size in top down step")

    ########## train
    train_args = parser.add_argument_group('training')

    train_args.add_argument("--dropout",
                            type=float, default=.1,
                            help="Global dropout rate")
    train_args.add_argument("--train_bsz",
                            type=int, default=512,
                            help="Batch size for training")
    train_args.add_argument("--n_epoch",
                            type=int, default=100,
                            help="Training epoches")
    train_args.add_argument("--clip_grad",
                            type=int, default=50,
                            help="Clip gradients to this value")
    train_args.add_argument("--loss_buf_sz",
                            type=int, default=20,
                            help="Buffer losses for the last m batches")
    #    train_args.add_argument("--train_from",
    #                            type=str,
    #                            help="Load trained model from (Replaced by model_load)")

    ## cosine annealing lr with restart
    train_args.add_argument("--lr_anr_type",
                            type=str, choices=["SGDR", "const"],
                            default="SGDR",
                            help='choose lr annealer in \
                            "cosine annealing with restart" | \
                            "constant lr (for hyperparams searching)" | \
                            ...')
    train_args.add_argument("--lr_start",
                            type=float, default=3 * 1e-4,
                            help="Initial and max lr in annealing")
    train_args.add_argument("--lr_end",
                            type=float, default=1e-6,
                            help="Final and min lr in annealing")
    train_args.add_argument("--lr_period",
                            type=int, default=10,
                            help="Epoches before next restart")
    train_args.add_argument("--lr_n_restarts",
                            type=int, default=5,
                            help="Times of restart in annealing")
    train_args.add_argument("--lr_mult_coeff",
                            type=int, default=1,
                            help="Mult coefficient for period increment")

    ## KL loss annealer: loss = beta * kl_loss + rec_loss with beta increasing from 0

    train_args.add_argument("--kl_anr_type",
                            type=str, choices=["linear_inc", "const", "cyclic", "expo"],
                            default="cyclic",
                            help='choose kl annealer in \
                            "linear increasing" | \
                            "constant lr" | \
                            "cyclical increasing" | \
                            ...')
    # linear/monotonic increasing annealer
    train_args.add_argument("--kl_e_start",
                            type=int, default=0,
                            help="Epoch start increasing KL weight")
    train_args.add_argument("--kl_w_start",
                            type=float, default=1e-4,
                            help="Initial KL weight")
    train_args.add_argument("--kl_w_end",
                            type=float, default=1e-3,
                            help="Final KL weight")
    # cyclical annealer
    train_args.add_argument("--kl_n_cycle",
                            type=int, default=1,
                            help="Cycles to repeat. Last cycle ends with `kl_w_end`")
    train_args.add_argument("--ratio",
                            type=float, default=0.2,
                            help="Propotion of a cycle is used for increasing beta")

    ########## sample
    sample_args = parser.add_argument_group('sampling')
    sample_args.add_argument("--sample_type",
                             type=str, choices=["control_z", "prior"],
                             default="prior",
                             help="Sample from a certain z or all zs")
    #    sample_args.add_argument("--n_sample",
    #                            type=int, default=1000,
    #                            help="Number of samples from prior distribution")
    sample_args.add_argument("--n_enc_zs",
                             type=int, default=[1000, 1000, 1000],
                             nargs="+",
                             help="Number of unique samples at each z layer (bottom z -> top z). \
                            For prior sampling, e.g. reversed([10,100,1000])")
    sample_args.add_argument("--n_dec_xs",
                             type=int, default=10,
                             help="n decoding attempts for each latent code")
    sample_args.add_argument("--gen_bsz",
                             type=int, default=128,
                             help="Batches when sampling in parallel")
    sample_args.add_argument("--max_len",
                             type=int, default=100,
                             help="Max length of sampled SMILES (includes bos and eos)")
    sample_args.add_argument("--sample_save",
                             type=str,
                             help="/dir/to/save/sample_smiles.csv")

    ## for control_z sampling
    sample_args.add_argument("--sample_layer",
                             type=int,
                             help="Index of z layer to control and sample")

    return parser

def GRAPH_LVAE(imput_dim, latent_dim, outputdim):
    parser = get_parser()
    config = parser.parse_args("--device cuda:0 \
                               --n_enc_zs 10 --n_dec_xs 10 --gen_bsz 128 \
                               --emb_sz 256 \
                               --enc_hidden_size 256 \
                               --enc_num_layers 1 \
                               --dec_hid_sz 512 \
                               --dec_n_layer 2 \
                               --ladder_d_size 256 128 64 \
                               --ladder_z_size 16 8 4 \
                               --ladder_z2z_layer_size 8 16 \
                               --dropout 0.2".split())
    model = LVAE(config)
    return model