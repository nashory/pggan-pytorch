import dataloader as DL
from config import config
import network as net
from math import floor, ceil

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm


class trainer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.nz = config.nz
        self.optimizer = config.optimizer

        self.resl = 2           # we start from 2^2 = 4
        self.max_resl = config.max_resl
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.globalIter = 0
        self.globalTick = 0
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen':None, 'dis':None}
        self.complete = {'gen':0, 'dis':0}
        self.phase = {'gen':'init', 'dis':'init'}
        self.flag_flush_gen = False
        self.flag_flush_dis = False
        self.trns_tick = 1
        self.stab_tick = 1
        
        # network and cirterion
        self.G = net.Generator(config)
        self.D = net.Discriminator(config)
        print ('Generator structure: ')
        print(self.G.model)
        print ('Discriminator structure: ')
        print(self.D.model)
        self.mse = torch.nn.MSELoss()
        if self.use_cuda:
            self.mse = self.mse.cuda()
            torch.cuda.manual_seed(config.random_seed)
            if config.n_gpu==1:
                #self.G = self.G.cuda()
                #self.D = self.D.cuda()         # It seems simply call .cuda() on the model does not function. PyTorch bug when we use modules.
                self.G = torch.nn.DataParallel(self.G).cuda(device_id=0)
                self.D = torch.nn.DataParallel(self.D).cuda(device_id=0)
            else:
                gpus = []
                for i  in range(config.n_gpu):
                    gpus.append(i)
                self.G = torch.nn.DataParallel(self.G, device_ids=gpus).cuda()
                self.D = torch.nn.DataParallel(self.D, device_ids=gpus).cuda()
        
        # define tensors, and get dataloader.
        self.renew_everything()

    def resl_scheduler(self):
        '''
        this function will schedule image resolution(self.resl) progressively.
        it should be called every iteration to ensure resl value is updated properly.
        step 1. (trns_tick) --> transition in generator.
        step 2. (stab_tick) --> stabilize.
        step 3. (trns_tick) --> transition in discriminator.
        step 4. (stab_tick) --> stabilize.
        '''
        if floor(self.resl) != 2 :
            self.trns_tick = self.config.trns_tick
            self.stab_tick = self.config.stab_tick
        
        self.batchsize = self.loader.batchsize
        delta = 1.0/(2*self.trns_tick+2*self.stab_tick)
        d_alpha = 1.0*self.batchsize/self.trns_tick/self.TICK

        # update alpha if fade-in layer exist.
        if self.resl%1.0 < (self.trns_tick)*delta:
            try:
                self.fadein['gen'].update_alpha(d_alpha)
                self.complete['gen'] = self.fadein['gen'].alpha*100
                self.phase['gen'] = 'gtrns'
            except:
                pass
        if self.resl%1.0 > (self.trns_tick+self.stab_tick)*delta and self.resl%1.0 < (self.trns_tick*2 + self.stab_tick)*delta:
            try:
                self.fadein['dis'].update_alpha(d_alpha)
                self.complete['dis'] = self.fadein['dis'].alpha*100
                self.phase['dis'] = 'dtrns'
            except:
                pass
        

        prev_kimgs = self.kimgs
        self.kimgs = self.kimgs + self.batchsize
        if (self.kimgs%self.TICK) < (prev_kimgs%self.TICK):
            self.globalTick = self.globalTick + 1

            # increase linearly every tick, and grow network structure.
            prev_resl = floor(self.resl)
            self.resl = self.resl + delta
            self.resl = max(2, min(10.5, self.resl))        # clamping, range: 4 ~ 1024

            # flush network.
            try:
                if self.resl%1.0 >= (self.trns_tick)*delta:
                    self.phase['gen'] = 'gstab'
                if self.flag_flush_gen and self.resl%1.0 >= (self.trns_tick + self.stab_tick)*delta:
                    self.G.module.flush_network()
                    self.flag_flush_gen = False
                    self.fadein['gen'] = None
                if self.resl%1.0 >= (2*self.trns_tick+self.stab_tick)*delta:
                    self.phase['dis'] = 'dstab'
                if floor(self.resl) != prev_resl:
                    self.D.module.flush_network()
                    self.flag_flush_dis = False
                    self.fadein['dis'] = None
            except:
                pass

            # grow network.
            if floor(self.resl) != prev_resl:
                self.G.module.grow_network(floor(self.resl))
                self.D.module.grow_network(floor(self.resl))
                self.renew_everything()
                self.fadein['gen'] = self.G.module.model.fadein_block
                self.fadein['dis'] = self.D.module.model.fadein_block
                self.flag_flush_gen = True
                self.flag_flush_dis = True

            
    def renew_everything(self):
        # renew dataloader.
        self.loader = DL.dataloader(config)
        self.loader.renew(floor(self.resl))
        
        # define tensors
        self.z = torch.FloatTensor(self.loader.batchsize, self.nz)
        self.z_test = torch.FloatTensor(self.loader.batchsize, self.nz)
        self.x = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.x_tilde = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.real_label = torch.FloatTensor(self.loader.batchsize).fill_(1)
        self.fake_label = torch.FloatTensor(self.loader.batchsize).fill_(0)

        # enable cuda
        if self.use_cuda:
            self.z = self.z.cuda()
            self.z_test = self.z_test.cuda()
            self.x = self.x.cuda()
            self.x_tilde = self.x.cuda()
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()
            torch.cuda.manual_seed(config.random_seed)

        # wrapping autograd Variable.
        self.x = Variable(self.x)
        self.x_tilde = Variable(self.x_tilde)
        self.z = Variable(self.z)
        self.z_test = Variable(self.z_test, volatile=True)
        self.real_label = Variable(self.real_label)
        self.fake_label = Variable(self.fake_label)
        
        # renew optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(self.G.parameters(), lr=self.config.lr, betas=betas, weight_decay=0.0)
            self.opt_d = Adam(self.D.parameters(), lr=self.config.lr, betas=betas, weight_decay=0.0)

    def feed_interpolated_input(self, x):
        alpha = self.complete['gen']/100.0
        transform = transforms.Compose( [   transforms.ToPILImage(),
                                            transforms.Scale(size=int(pow(2,floor(self.resl)-1)), interpolation=0),      # 0: nearest
                                            transforms.Scale(size=int(pow(2,floor(self.resl))), interpolation=0),      # 0: nearest
                                        ] )
        if self.phase == 'gtrns' and floor(self.resl)>2:
            x_intp = torch.add(x.mul(alpha), transform(x).mul(1-alpha))
            return x_intp
        else:
            return x


    def train(self):
        for step in range(2, self.max_resl):
            for iter in tqdm(range(0,(self.trns_tick*2+self.stab_tick*2)*self.TICK, self.loader.batchsize)):
                self.globalIter = self.globalIter+1
                self.stack = self.stack + self.loader.batchsize
                if self.stack > ceil(len(self.loader.dataset)):
                    self.epoch = self.epoch + 1
                    self.stack = self.stack%(ceil(len(self.loader.dataset)))

                # reslolution scheduler.
                self.resl_scheduler()
                
                # zero gradients.
                self.G.zero_grad()
                self.D.zero_grad()

                # update discriminator.
                #self.x.data = self.feed_interpolated_input(self.loader.get_batch())
                #self.x.data = self.loader.get_batch()
                self.x = self.loader.get_batch()
                self.x = Variable(self.x)


                self.z.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
                self.x_tilde = self.G(self.z)
               
                fx = self.D(self.x)
                fx_tilde = self.D(self.x_tilde.detach())
                loss_d = self.mse(fx, self.real_label) + self.mse(fx_tilde, self.fake_label)
                loss_d.backward()
                self.opt_d.step()

                # update generator.
                fx_tilde = self.D(self.x_tilde)
                loss_g = self.mse(fx_tilde, self.real_label.detach())
                loss_g.backward()
                self.opt_g.step()

                # logging.
                log_msg = '[E:{0}][T:{1}][{2:6}/{3:6}]  errD: {4:.4f} | errG: {5:.4f} | [cur:{6:.3f}][resl:{7:4}][{8}/{9:.1f}%][{10}/{11:.1f}%]'.format(self.epoch, self.globalTick, self.stack, len(self.loader.dataset), loss_d.data[0], loss_g.data[0], self.resl, int(pow(2,floor(self.resl))), self.phase['gen'], self.complete['gen'], self.phase['dis'], self.complete['dis'])
                tqdm.write(log_msg)



## perform training.
print '----------------- configuration -----------------'
for k, v in vars(config).items():
    print('  {}: {}').format(k, v)
print '-------------------------------------------------'
torch.backends.cudnn.benchmark = True           # boost speed.
trainer = trainer(config)
trainer.train()


