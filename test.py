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
print(G.model)
G.grow_network(3)
print(G.model)
G.flush_network()
print(G.model)




