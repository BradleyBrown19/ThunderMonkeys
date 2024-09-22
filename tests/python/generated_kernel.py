
import torch 
import torch.nn as nn
import thunderkittens as tk

TOL = 1e-2

def rtol(x, y): return 2 * (x - y).abs() / (x.abs() + y.abs())

def run_reference_implementation(x, residual, drop_path, dropout, norm, residual_in_fp32=False):
    dropped = dropout(x) #drop_path(dropout(x))
    residual = (residual + dropped ) if residual is not None else dropped
    out = norm(residual.to(dtype=norm.weight.dtype))
    residual = residual.to(torch.float32)
    return out, residual

def run_tk(x, residual, drop_path, dropout, norm, residual_in_fp32=False):
    x = x.to(dtype=torch.bfloat16)
    residual = residual.to(dtype=torch.bfloat16)
    norm_weight = norm.weight.to(dtype=torch.bfloat16)
    norm_bias = norm.bias.to(dtype=torch.bfloat16)

    has_residual = int(residual is not None)
    out = torch.zeros_like(x)
    out_resid = torch.zeros_like(x)

    tk.fused_layernorm(
        int(has_residual), float(dropout.p),
        x, residual, 
        norm_weight, norm_bias, 
        out, out_resid
    )

    return out, out_resid 

if __name__ == "__main__":

    b, n, d = 16, 32, 1024
    p = 0.0
    p_path = 0.00

    torch.manual_seed(0)
    x = torch.randn((b, n, d), device='cuda')
    residual = torch.randn((b, n, d), device='cuda')

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = None
    out, resid = run_reference_implementation(x, residual, drop_path, dropout, norm)

    outs = []
    resids = []
    
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = None 
    fn_out, fn_resid = run_tk(x, residual, drop_path, dropout, norm)

    print("----"*10)
    diff = rtol(out, fn_out).mean().item()
    assert diff < TOL
    print(f"Out Diff: {diff}")

    diff = rtol(resid, fn_resid).mean().item() 
    assert diff < TOL
    print(f"Resid Diff: {diff}")