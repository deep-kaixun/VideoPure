import torch
from einops import rearrange
import torch.nn.functional as F


class Flow_models(torch.nn.Module):
    def __init__(self,flow_model,flow_t=1,flow_s=-4,w=800,w1=5):
        super().__init__()
        self.flow_model=flow_model
        self.flow_model.cuda()
        self.flow_model.eval()
        self.flow_t=flow_t
        self.flow_s=flow_s
        self.tt=0
        self.w=w
        self.w1=w1
    def forward(self,x):
        return self.flow_model(x)
    
    @torch.no_grad()
    def compute_flows(self,videos):
        videos=videos.permute(0,2,1,3,4)
        flows = [self.compute_flow(videos.clone())]
        flows = [rearrange(flow, 'b t c h w -> (b t) c h w') for flow in flows]
        flows = [self.resize_flow(flow, size_type='ratio', sizes=(0.125, 0.125)) for flow in flows]
        flows = [rearrange(flow, '(b t) c h w -> b t c h w', t=videos.size(1)-1) for flow in flows]
        return flows[0]
    

   
    
    def compute_temporal_condition_v4(self, flows, latents, masks):
        
        flow_fwd_prop = flows
        t = latents.size(0)
        # b,t,c,h//8,w//8
        latents = rearrange(latents, '(b t) c h w -> b t c h w', t=t)
        loss_f = 0
        latent_curr_warp = latents[:, 0, :, :, :]
        # forward propagation
        for i in range(0, t):
            latent_curr = latents[:, i, :, :, :]
            # return latent_curr
            if i > 0:
                flow = flow_fwd_prop[:, i - 1, :, :, :]
                latent_curr_warp = self.flow_warp(latent_curr.clone(), flow.permute(0, 2, 3, 1), interp_mode='bilinear')
                loss_f += F.l1_loss(latent_curr_warp, latent_prev)
            latent_prev = latent_curr
        return loss_f


    @torch.no_grad()
    def flow_warp(self,x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, return_mask=False):
        assert x.size()[-2:] == flow.size()[1:3]
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False

        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
        
        # TODO, what if align_corners=False
        if not return_mask:
            return output

        return output

    
    @torch.no_grad()
    def latents_optimize_with_flow(self,latents,flows,masks,model_log_variance,guidance_scale=-5,ori_latents=None,beta_prod_t=None,alpha_prod_t=None,noise_pred=None,flag=0):

        step=self.flow_t
        guidance_scale=(self.flow_s)/max(step,1)
        latents=latents.detach().float()
        bsz, channel, frames, width, height = ori_latents.shape
        ori_latents = ori_latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height).float()
        loss_tempo = 0
        loss_mse=0.
        for _ in range(step):
            with torch.enable_grad():
                    latents = latents.detach()
                    latents.requires_grad = True
                    if noise_pred is not None:
                        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                    else:
                        pred_original_sample=latents
                    loss_mse=F.mse_loss(pred_original_sample,ori_latents)
                    loss_tempo = -self.compute_temporal_condition_v4(flows, pred_original_sample, masks)
                    loss=(self.w1*loss_tempo+self.w*loss_mse)
                    latents = latents + guidance_scale * model_log_variance * torch.autograd.grad(loss, latents)[0]
                    latents = latents.detach()
        return latents.detach().half()
    
    
    
    
    # @torch.no_grad()
    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # former
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)   # latter
        flows_backward = self.flow_model(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        return flows_backward
    @torch.no_grad()
    def resize_flow(self,flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
        _, _, flow_h, flow_w = flow.size()
        if size_type == 'ratio':
            output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
        elif size_type == 'shape':
            output_h, output_w = sizes[0], sizes[1]
        else:
            raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')
        input_flow = flow.clone()
        ratio_h = output_h / flow_h
        ratio_w = output_w / flow_w
        input_flow[:, 0, :, :] *= ratio_w
        input_flow[:, 1, :, :] *= ratio_h
        resized_flow = F.interpolate(
            input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
        return resized_flow
    
