from tqdm import tqdm
from roma.utils.utils import to_cuda
import roma
import torch
import wandb
import torch.nn.functional as F
#from torchviz import make_dot

def log_param_statistics(named_parameters, norm_type = 2):
    named_parameters = list(named_parameters)
    grads = [p.grad for n, p in named_parameters if p.grad is not None]
    weight_norms = [p.norm(p=norm_type) for n, p in named_parameters if p.grad is not None]
    names = [n for n,p in named_parameters if p.grad is not None]
    param_norm = torch.stack(weight_norms).norm(p=norm_type)
    device = grads[0].device
    grad_norms = torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads])
    nans_or_infs = torch.isinf(grad_norms) | torch.isnan(grad_norms)
    nan_inf_names = [name for name, naninf in zip(names, nans_or_infs) if naninf]
    total_grad_norm = torch.norm(grad_norms, norm_type)
    if torch.any(nans_or_infs):
        print(f"These params have nan or inf grads: {nan_inf_names}")
    

def train_step(train_batch, model, objective, optimizer, grad_scaler, grad_clip_norm = 1.,**kwargs):
    optimizer.zero_grad()
    out = model(train_batch)
    l = objective(out, train_batch)

    #make_dot(l, params=dict(list(model.named_parameters()))).render(f"./logs/rnn_torchviz_{roma.GLOBAL_STEP}", format="png")

    grad_scaler.scale(l).backward()
    grad_scaler.unscale_(optimizer)
    log_param_statistics(model.named_parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm) # what should max norm be?
    grad_scaler.step(optimizer)
    grad_scaler.update()
    
    if grad_scaler._scale < 1.:
        grad_scaler._scale = torch.tensor(1.).to(grad_scaler._scale)
    roma.GLOBAL_STEP = roma.GLOBAL_STEP + roma.STEP_SIZE # increment global step
    return {"train_out": out, "train_loss": l.item()}


def train_k_steps(
    n_0, k, dataloader, model, objective, optimizer, lr_scheduler, grad_scaler, progress_bar=True, grad_clip_norm = 1., warmup = None, ema_model = None,
):
    for n in tqdm(range(n_0, n_0 + k), disable=(not progress_bar) or roma.RANK > 0):
        batch = next(dataloader)
        model.train(True)
        batch = to_cuda(batch)
        train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
            n=n,
            grad_clip_norm = grad_clip_norm,
        )
        if ema_model is not None:
            ema_model.update()
        if warmup is not None:
            with warmup.dampening():
                lr_scheduler.step()
        else:
            lr_scheduler.step()
        

def get_imA_transfer(corresps, batch):
    finest_scale = 1
    im_A_to_im_B = corresps[finest_scale]["flow"].permute(0, 2, 3, 1)
    certainty = corresps[finest_scale]["certainty"]

    weight = certainty.sigmoid().detach()  # logits -> probs
    #weight = torch.ones_like(certainty)

    if (im_A_to_im_B.abs() > 1).any() and True:
        wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
        weight[wrong[:,None]] = 0
    im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)

    x_B_norm = batch['im_B']

    im_A_transfer = F.grid_sample(x_B_norm, im_A_to_im_B, mode="bilinear", align_corners=False)
    
    return im_A_transfer, weight

def get_im_transfer(corresps, batch, symmetric = False):
    finest_scale = 1
    im_A_to_im_B = corresps[finest_scale]["flow"].permute(0, 2, 3, 1)
    certainty = corresps[finest_scale]["certainty"]

    weight = certainty.sigmoid().detach()  # logits -> probs
    #weight = torch.ones_like(certainty)

    if (im_A_to_im_B.abs() > 1).any() and True:
        wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
        weight[wrong[:,None]] = 0
    im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
    
    x_A_norm = batch['im_A']
    x_B_norm = batch['im_B']

    if symmetric:
        
        A_to_B, B_to_A = im_A_to_im_B.chunk(2)
        im_A_transfer = F.grid_sample(x_B_norm, A_to_B, mode="bilinear", align_corners=False)
        im_B_transfer = F.grid_sample(x_A_norm, B_to_A, mode="bilinear", align_corners=False)
        
        return im_A_transfer, im_B_transfer, weight
    else:
        im_A_transfer = F.grid_sample(x_B_norm, im_A_to_im_B, mode="bilinear", align_corners=False)
        return im_A_transfer, None, weight

def train_step_ssl0(train_batch, model, objective, optimizer, grad_scaler, grad_clip_norm = 1., symmetric = False, **kwargs):
    with torch.autograd.set_detect_anomaly(False):

        optimizer.zero_grad()
        corresps = model(train_batch)
        #loss_sup = objective(corresps, train_batch)
        #loss_sup = 0
        
        batch_imA_transfer = {}
        batch_imA_transfer['im_A'], _ = get_imA_transfer(corresps, train_batch)
        batch_imA_transfer['im_B'] = train_batch['im_A']

        corresps_imA_transfer = model(batch_imA_transfer)
        
        im_A_pred,  weight = get_imA_transfer(corresps_imA_transfer, batch_imA_transfer)
        loss_ssl = objective(train_batch['im_A'], im_A_pred, weight)
        
        #l = loss_sup + loss_ssl
        l = loss_ssl
        #make_dot(l, params=dict(list(model.named_parameters()))).render(f"./logs/rnn_torchviz_{roma.GLOBAL_STEP}", format="png")

        grad_scaler.scale(l).backward()
        grad_scaler.unscale_(optimizer)
        log_param_statistics(model.named_parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm) # what should max norm be?
        grad_scaler.step(optimizer)
        grad_scaler.update()
        
        if grad_scaler._scale < 1.:
            grad_scaler._scale = torch.tensor(1.).to(grad_scaler._scale)
        roma.GLOBAL_STEP = roma.GLOBAL_STEP + roma.STEP_SIZE # increment global step
        return {"train_out": corresps, "train_loss": l.item()}

def train_step_ssl(train_batch, model, objective, optimizer, grad_scaler, grad_clip_norm = 1., symmetric = False, **kwargs):
    with torch.autograd.set_detect_anomaly(False):

        optimizer.zero_grad()
        corresps = model(train_batch)
        #loss_sup = objective(corresps, train_batch)
        #loss_sup = 0
        
        if symmetric:
            #print('computing symmetric ssl loss')
            batch_transfer = {}
            im_A_transfer, im_B_transfer, weight = get_im_transfer(corresps, train_batch, symmetric=symmetric)

            batch_transfer['im_A'] = im_A_transfer
            batch_transfer['im_B'] = im_B_transfer

            corresps_transfer = model(batch_transfer)
            im_A_pred, im_B_pred, weight = get_im_transfer(corresps_transfer, batch_transfer, symmetric=symmetric)
            weight_A, weight_B = weight.chunk(2)
            loss_ssl = objective(train_batch['im_A'], im_A_pred, weight_A) + \
                       objective(train_batch['im_B'], im_B_pred, weight_B)
        else:
            batch_imA_transfer = {}
            im_A_transfer, _ = get_imA_transfer(corresps, train_batch)
            batch_imA_transfer['im_A'] = im_A_transfer
            batch_imA_transfer['im_B'] = train_batch['im_A']

            corresps_imA_transfer = model(batch_imA_transfer)
            
            im_A_pred,  weight = get_imA_transfer(corresps_imA_transfer, batch_imA_transfer)
            loss_ssl = objective(train_batch['im_A'], im_A_pred, weight)
        
        #l = loss_sup + loss_ssl
        l = loss_ssl
        #make_dot(l, params=dict(list(model.named_parameters()))).render(f"./logs/rnn_torchviz_{roma.GLOBAL_STEP}", format="png")

        grad_scaler.scale(l).backward()
        grad_scaler.unscale_(optimizer)
        log_param_statistics(model.named_parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm) # what should max norm be?
        grad_scaler.step(optimizer)
        grad_scaler.update()
        
        if grad_scaler._scale < 1.:
            grad_scaler._scale = torch.tensor(1.).to(grad_scaler._scale)
        roma.GLOBAL_STEP = roma.GLOBAL_STEP + roma.STEP_SIZE # increment global step
        return {"train_out": corresps, "train_loss": l.item()}

def train_k_steps_ssl(
    n_0, k, dataloader, model, objective, optimizer, lr_scheduler, grad_scaler, progress_bar=True, grad_clip_norm = 1., warmup = None, ema_model = None, symmetric = False
):
    for n in tqdm(range(n_0, n_0 + k), disable=(not progress_bar) or roma.RANK > 0):
        batch = next(dataloader)
        model.train(True)
        batch = to_cuda(batch)
        train_step_ssl(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
            n=n,
            grad_clip_norm = grad_clip_norm,
            symmetric = symmetric,
        )
        if ema_model is not None:
            ema_model.update()
        if warmup is not None:
            with warmup.dampening():
                lr_scheduler.step()
        else:
            lr_scheduler.step()
        

def train_epoch(
    dataloader=None,
    model=None,
    objective=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
):
    model.train(True)
    print(f"At epoch {epoch}")
    for batch in tqdm(dataloader, mininterval=5.0):
        batch = to_cuda(batch)
        train_step(
            train_batch=batch, model=model, objective=objective, optimizer=optimizer
        )
    lr_scheduler.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": epoch,
    }


def train_k_epochs(
    start_epoch, end_epoch, dataloader, model, objective, optimizer, lr_scheduler
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_epoch(
            dataloader=dataloader,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
        )
