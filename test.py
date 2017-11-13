import dataloader as DL
from config import config
import network as net
from math import floor, ceil

import torch
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm

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
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.globalIter = 0
        
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
            self.z.cuda()
            self.z_test.cuda()
            self.x.cuda()
            self.real_label.cuda()
            self.fake_label.cuda()
            self.mse = mse.cuda()
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            torch.cuda.manual_seed(config.random_seed)




    def train(self):
   
        print(self.G.model)
        print(self.D.model)

        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            opt_g = Adam(self.G.parameters(), lr=self.config.lr, betas=betas, weight_decay=0.0)
            opt_d = Adam(self.D.parameters(), lr=self.config.lr, betas=betas, weight_decay=0.0)

        
        '''
        self.z = Variable(self.z)
        self.x = Variable(self.x)
        self.z_test = Variable(self.z_test, volatile=True)
        self.real_label = Variable(self.real_label)
        self.fake_label = Variable(self.fake_label)
        '''
        self.z = Variable(self.z)
        self.z_test = Variable(self.z_test, volatile=True)
        self.real_label = Variable(self.real_label)
        self.fake_label = Variable(self.fake_label)


        for step in range(2, self.max_resl):
            for iter in tqdm(range(0,(self.trns_tick+self.stab_tick)*self.TICK*2, self.loader.batchsize)):
                self.globalIter = self.globalIter+1


                # zero gradients.
                self.G.zero_grad()
                self.D.zero_grad()

                # update discriminator.
                self.x = self.loader.get_batch()
                self.x = Variable(self.x)
                self.z.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
                self.x_tilde = self.G(self.z)
               
                fx = self.D(self.x)
                fx_tilde = self.D(self.x_tilde.detach())
                loss_d = self.mse(fx, self.real_label) + self.mse(fx_tilde, self.fake_label)
                loss_d.backward()
                opt_d.step()

                # update generator.
                fx_tilde = self.D(self.x_tilde)
                loss_g = self.mse(fx_tilde, self.real_label.detach())
                loss_g.backward()
                opt_g.step()




trainer = trainer(config)
trainer.train()



