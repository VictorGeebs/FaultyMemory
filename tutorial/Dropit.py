import torch
import torch.nn as nn
TEMPERATURE = 2/3
CONSTANT = 12.8

# Dim : other, self
XOR_GRADS = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [-1, 1, 1, 1, 1, 1, 1, -1],
    [1, -1, -1, 1, 1, -1, -1, 1],
    [-1, -1, -1, 1, 1, -1, -1, -1],
    [1, 1, 1, -1, -1, 1, 1, 1],
    [-1, 1, 1, -1, -1, 1, 1, -1],
    [1, -1, -1, -1, -1, -1, -1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1]
]

# Portemanteau of Dropout and Bit
class Dropit(nn.Module):
    def __init__(self, precision=3, initial_p=0.01, trainable=False, use_oldXOR=False):
        super().__init__()
        if precision != 3:
            raise ValueError("Only precision 3 is supported")
        self.precision = precision
        self.trainable = trainable
        self.register_buffer('uses', torch.Tensor([0]))
        self.register_buffer('fan_in', torch.Tensor([0]))
        self.register_buffer('constant', torch.Tensor([CONSTANT]))
        self.register_buffer('temp', torch.Tensor([TEMPERATURE]))
        self.register_buffer('bitrank', torch.tensor([1, 2, 4]))
        # self.register_buffer('XOR_GRAD', torch.FloatTensor([7, 5, 3, 1, -1, -3, -5, -7]) -> would cause gradient explosion !
        if use_oldXOR: self.register_buffer('XOR_GRAD', torch.FloatTensor([1, 1, 1, 1, -1, -1, -1, -1]))
        else: self.register_buffer('XOR_GRAD', torch.FloatTensor(XOR_GRADS))
        self.use_oldXOR = use_oldXOR
        self.register_parameter('dropitp', nn.Parameter(
            torch.Tensor([initial_p]*precision)))
        self.dropitp.requires_grad = trainable
        p = getattr(self, f'dropitp')
        p.ptype = self.fan_in
        p.pact = True

    def forward(self, x):
        compact_bernouilli = MyRelaxedBernoulli(self.temp, probs=self.dropitp)
        compact_sample = compact_bernouilli.rsample(
            sample_shape=x.size()).squeeze_(dim=-1)
        # if self.use_oldXOR: 
        #     compact_h = torch.sum(compact_sample * self.bitrank, dim=-1)
        #     res = old_xored(x, compact_h, self.XOR_GRAD)
        # else: 
        compact_sample = torch.sum(compact_sample * self.bitrank, dim=-1)
        res = xored(x, compact_sample)
        return res

    def update_fan_in(self, qty):
        self.fan_in += qty
        self.uses += 1

    def reset_fan_in(self):
        self.fan_in = torch.zeros_like(self.fan_in)
        self.uses = torch.zeros_like(self.uses)

    def consumption(self):
        return torch.clamp(- torch.log(self.dropitp)/self.constant, 0, 1) * self.fan_in

    def maxcons(self):
        return self.fan_in*len(self.dropitp)
    
    # def get_consumption(self):
    #     return clampzeroone(-torch.log(self.dropitp)/self.constant) * self.fan_in

class MyRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx,grad):
        return grad
rounded = MyRound.apply

class MyXOR(torch.autograd.Function):
    """ XOR ops
    Few things to note :
    - need to round the input because its cast to int8 for the xor op -- may severly damage performance if not rounded !
    - input/xormask are cast to uint8 -- use this after Relu6/quantized ops ! :) 
    - XOR_grad should be the sign of the analytic gradient, otherwise grad explosion through the net
    """
    @staticmethod
    def forward(ctx,input_raw,xormask):
        input_int = torch.round(input_raw).to(dtype=torch.uint8) # round != cast to int
        xormask_int = xormask.to(dtype=torch.uint8)
        ctx.save_for_backward(input_int, xormask_int)
        res = torch.bitwise_xor(input_int, xormask_int).to(dtype=input_raw.dtype)
        return res
    
    @staticmethod
    def backward(ctx,grad):
        return grad, None

xored = MyXOR.apply

class MyRelaxedBernoulli(torch.distributions.RelaxedBernoulli):
    def rsample(self,*args,**kwargs):
        sample = super(MyRelaxedBernoulli,self).rsample(*args,**kwargs)
        return rounded(sample)

# class DropitFunction(torch.autograd.Function):
#     """
#     Dropit function for Relu activation (clamp between zero and 2^precision)
#     """
#     @staticmethod
#     def forward(ctx, rng_states, precision, ps, input):
#         input = torch.round(torch.clamp(input.detach(), 0, 2 ** precision))
#         if ps.any(): # only do the long things if there is a need for it
#             dims = input.size()
#             input = torch.flatten(input)
#             dropit_numba[BLOCKS, THREADS_PER_BLOCK](rng_states, ps, input, precision, torch.zeros_like(input))
#             input = torch.reshape(input, dims)
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Straight through gradient flow. The memory error is not propagated to the inner gradient descent
#         """
#         return None, None, None, grad_output.clone() # number of arg after ctx in forward

# dropit = DropitFunction.apply

# def get_faulty_actimem(model, seed = 42):
#     """
#     Ensure the \bm{p} of all DropitModule inside a nn.Module are shared (aligned on the first DropitModule)
#     Return a \bm{p} given the quantization scheme controlling all the DropitModule
#     Initiatilise the random state for Numba
#     """
#     ps, prec = None, None
#     rng_state = create_xoroshiro128p_states(THREADS_PER_BLOCK * BLOCKS, seed=seed)
#     for m in model.modules():
#         if isinstance(m, Dropit):
#             if ps is None: ps = m.ps; prec = m.precision
#             else: m.ps = ps
#             setattr(m, 'rng_states', rng_state)
#     print(f'prec {prec} - initial p {ps} - returned p {ps[-prec:]}')
#     return ps[-prec:] # np slices create a view

# def check_faulty_actimem(model):
#     """
#     Assert the values or Dropit nn.Module are still aligned
#     """
#     ps = None
#     for m in model.modules():
#         if isinstance(m, Dropit):
#             if ps is None: ps = m.ps
#             assert np.equal(ps, m.ps).all()
#     assert ps is not None

# def print_faulty_actimem(model):
#     for m in model.modules():
#         if isinstance(m, Dropit):
#             #TODO
#             pass

# ##
# # NUMBA SOLUTION ?
# # PROS : clear, up to date, concise code, abstract CUDA's management
# ##
# @cuda.jit
# def dropit_numba(rng_states, ps, in_val, prec, done_mask):
#     thread_id = cuda.grid(1)
#     stride_x= cuda.gridsize(1)
#     for x in range(thread_id, in_val.shape[0], stride_x):
#         # temp = in_val[x]
#         inv_mask = 0
#         for i in range(prec):
#             inv_mask += (xoroshiro128p_uniform_float32(rng_states, thread_id) < ps[7-i]) * 2 ** i ### READ FROM RIGHT TO LEFT
#         if done_mask[x] == 0:
#             in_val[x] = int(inv_mask) ^ int(in_val[x])
#             done_mask[x] = 1
#         # print(int(inv_mask), ' xor ', int(temp), ' = ', in_val[x])

# ##
# # COMPILED KERNEL SOLUTION ?
# # CONS : seems old, no much ressources on this
# ##
# kernel = '''
# extern "C"

# __global__ void bitwise_xor(int *input, bool mask[])
# {
#     int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if(i >= total)
#         return;
#     input[i] = input[i] ^ mask[];
# }
# '''

# if __name__ == '__main__':
#     print("Testing Dropit module")
#     ##
#     # Working condition
#     ##
#     original = torch.rand((3,3)).cuda()*6
#     print(original)
#     p_test = np.linspace(0,1,3)
#     print(p_test)
#     drop_module = Dropit(p=p_test)
#     drop_module.ps[7] = 1
#     get_faulty_actimem(drop_module)
#     res = drop_module.forward(original.clone())
#     print(res)
#     assert not torch.equal(res, original)

#     ##
#     # Test for no changes
#     # FIXME not working with quantization inside the module
#     ##
#     drop_module = Dropit(p=0)
#     get_faulty_actimem(drop_module)
#     res = drop_module.forward(original)
#     #assert torch.equal(res, original)
