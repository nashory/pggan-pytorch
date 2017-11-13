import dataloader as DL
from config import config
import network as net
from math import floor, ceil

import torch
from torch.optim import Adam

# dataloader test.
#loader = DL.dataloader(config)
#loader.renew(config)



# model loading test.
#G = net.Generator(config)
#print(G.first_block())
#print(G.intermediate_block(6))
#print(G.intermediate_block(7))
#print(G.intermediate_block(8))
#print(G.intermediate_block(9))
#print(G.intermediate_block(10))
#a, ndim = G.intermediate_block(10)
#b = G.to_rgb_block(ndim)
#print(a)
#print(b)

class trainer:
    def __init__(self, config):
        self.config = config
        self.nz = config.nz
        self.optimizer = config.optimizer

        self.resl = 2           # we start from 2^2 = 4
        self.max_resl = config.max_resl
        
        # dataloader
        self.loader = DL.dataloader(config)
        #self.loader.renew(self.resl)
        
        # network and cirterion
        self.G = net.Generator(config)
        self.D = net.Discriminator(config)
        self.mse = torch.nn.MSELoss()

        # define tensors
        self.z = torch.FloatTensor(self.loader.batchsize, self.nz)
        self.z_test = torch.FloatTensor(self.loader.batchsize, self.nz)
        self.x = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.real_label = torch.FloatTensor(self.loader.batchsize).fill_(1)
        self.fake_label = torch.FloatTensor(self.loader.batchsize).zero_()

        # enable cuda
        if config.use_cuda:
            z.cuda()
            z_test.cuda()
            x.cuda()
            real_label.cuda()
            fake_label.cuda()
            mse = mse.cuda()
            torch.cuda.manual_seed(config.random_seed)




    def train(self):
   
        # network
        #G = net.Generator(config)
        #print(G.model)
        #G.grow_network(3)
        #print(G.model)
        #G.flush_network()
        #print(G.model)
    
        # tensor

        #self.G.grow_network(3)
        print(self.G.model)
        print(self.D.model)

        #self.x = self.loader.get_batch()



        #data = iter(self.loader)
        print len(self.loader)
        #self.x = data.next()


        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            opt_g = Adam(self.G.parameters(), lr=self.config.lr, betas=betas, weight_decay=0.0)


            #opt_d = Adam(self.D.parameters(), lr=self.config.lr, betas=betas, weight_decay=0.0)

        



trainer = trainer(config)
trainer.train()



