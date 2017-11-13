import dataloader as DL
from config import config
import network as net

# dataloader test.
#loader = DL.dataloader(config)
#loader.renew(config)



# model loading test.
G = net.Generator(config)
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


# test fadein layer.
model = G.get_init_gen()
a = G.grow_network(model, 3)
print(a)





