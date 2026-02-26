import torch


class Config: 


    # def __init__(self, block_size=128, batch_size=16, grad_accum_steps=8, n_layer=6, n_embd=192, 
    #              n_head=6, dropout=0.1, learning_rate=3e-4, max_iters=30_000, 
    #              eval_interval=1_000, eval_iters=200, checkpoint_interval=10_000):
        
    #     self.block_size = block_size
    #     self.batch_size = batch_size
    #     self.grad_accum_steps = grad_accum_steps
    #     self.n_layer = n_layer
    #     self.n_embd = n_embd
    #     self.n_head = n_head
    #     self.dropout = dropout
    #     self.learning_rate = learning_rate
    #     self.max_iters = max_iters
    #     self.eval_interval = eval_interval
    #     self.eval_iters = eval_iters
    #     self.CHECKPOINT_INTERVAL = checkpoint_interval

    # SYSTEM 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = (device == "cuda")
    num_workers = 0
 
    # OK - 5M params , 80M tokens 
    # 1 test overfitting 
    
    # block_size = 128
    # batch_size = 8
    # grad_accum_steps = 4
    
    # n_layer = 4
    # n_embd = 128
    # n_head = 4
    
    # dropout = 0.1
    
    # max_lr = 5e-4 
    # min_lr = 5e-5
    # warmup_iters = 1000
    # lr_decay_iters = 15_000   
    # max_iters = 15_000
    
    # eval_interval = 500
    # eval_iters = 100
    
    # CHECKPOINT_INTERVAL = 5_000
 
#  ====================================================================================================

    # MID - 12M params, 490M tokens 
    # 1 real training
    # 2 stable convergence
    # 3 good quality vs cost  

    # block_size = 128
    # batch_size = 16
    # grad_accum_steps = 8

    # n_layer = 6
    # n_embd = 192
    # n_head = 6 
    # dropout = 0.1

    # max_lr = 4e-4
    # min_lr = 4e-5 
    # warmup_iters = 2000
    # lr_decay_iters = 30_000    
    # max_iters = 30_000

    # eval_interval = 1_000
    # eval_iters = 200

    # CHECKPOINT_INTERVAL = 10_000

# ====================================================================================================
 
    # BETTER - 45M params, 1B+ tokens 
    # 1 you have more data
    # 2  more VRAM
    # 3 longer training budget 

    block_size = 256
    batch_size = 16
    grad_accum_steps = 8
    
    n_layer = 12
    n_embd = 384
    n_head = 6
    
    dropout = 0.1
    max_lr = 2e-4
    min_lr = 2e-5
    warmup_iters = 4_000
    lr_decay_iters = 50_000
    max_iters = 50_000
    
    eval_interval = 1_000
    eval_iters = 300
    
    CHECKPOINT_INTERVAL = 25_000


# ====================================================================================================
    # FIN_TUNING CONFIG
    # BETTER - 45M params, 1B+ tokens 
    # 1 you have more data
    # 2  more VRAM
    # 3 longer training budget 

    # block_size = 192
    # batch_size = 4
    # grad_accum_steps = 8
    
    # n_layer = 12
    # n_embd = 384
    # n_head = 6
    
    # dropout = 0.1
    # max_lr = 2e-4
    # min_lr = 2e-5
    # warmup_iters = 4_000
    # lr_decay_iters = 50_000
    # max_iters = 50_000
    
    # eval_interval = 1_000
    # eval_iters = 300
    
    # CHECKPOINT_INTERVAL = 25_000
