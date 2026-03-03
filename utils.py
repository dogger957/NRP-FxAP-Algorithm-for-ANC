import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.distributed as dist
def is_dist_avail_and_initialized():
    """ Check if distributed environment is supported. """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # for single GPU
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value

def slicing(x, slice_idx, axes):
    dimensionality = len(x.shape) 
    if dimensionality == 3:
       if axes == 1:
           return x[:,slice_idx,:]
       if axes == 2:
           return x[:,:,slice_idx]
    if dimensionality == 2:
       if axes == 0:
           return x[slice_idx,:]
       if axes == 1:
           return x[:,slice_idx]
       


def SEF(y, etafang, num_points=1000):

    if y.dim() == 1:
        y = y.unsqueeze(0)  
    
    if etafang == 0:
        return y.unsqueeze(1)    # linear

    sign = torch.sign(y)
    y_abs = torch.abs(y)
    y_abs = torch.nan_to_num(y_abs, nan=0.0, posinf=1e6, neginf=-1e6)
    etafang = max(etafang, 1e-6)
    z = torch.linspace(0, 1, num_points, device=y.device, dtype=y.dtype).view(1, 1, -1)
    z = z * y_abs.unsqueeze(-1)
    integrand = torch.exp(-z**2 / (2 * etafang))
    # Check for numerical issues
    if torch.isnan(integrand).any() or torch.isinf(integrand).any():
        print(f"NaN/Inf detected in integrand! etafang={etafang}")

    dz = y_abs / (num_points - 1)
    dz = torch.where(dz == 0, torch.tensor(1e-6, device=dz.device, dtype=dz.dtype), dz)

    integral_abs = torch.sum((integrand[:, :, 1:] + integrand[:, :, :-1]) / 2, dim=2) * dz

    integral = sign * integral_abs
    return integral.unsqueeze(1)  


