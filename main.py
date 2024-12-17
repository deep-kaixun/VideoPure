import argparse
import random
import torch
from pipe_defense import Video_Diffpure
from datasets import get_loaders
from load_models import load_classifier
from tqdm import tqdm
from schedule.scheduling_ddim import DDIMScheduler
from schedule.scheduling_ddpm import DDPMScheduler
import argparse
from core.raft_arch import RAFT_SR
from flow_net import Flow_models
import os
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t',default=5,type=int,help='denoising timestep')
    parser.add_argument('--noise_type',default='videopure',type=str,help='use small model')
    parser.add_argument('--attack_method',default='pgd',type=str,help='use small model')
    parser.add_argument('--attack_iter',default=10,type=int,help='use small model')
    parser.add_argument('--eps',default=4/255,type=float,help='use small model')
    parser.add_argument('--attack_alpha',default=2/255,type=float,help='use small model')
    parser.add_argument('--attack_device',default='cuda',type=str,help='use small model')
    parser.add_argument('--classfier',default='NL_res50',type=str,help='use small model')
    parser.add_argument('--ckpt',default='ckpt',type=str,help='use small model')
    parser.add_argument('--model',default='damo-vilab/text-to-video-ms-1.7b',type=str,help='use small model')
    parser.add_argument('--flow_model',default='raft',type=str,help='use small model')
    parser.add_argument('--flow_model_path',default='raft-things.pth',type=str,help='use small model')
    parser.add_argument('--datasets',default='UCF101',type=str,help='use small model')
    parser.add_argument('--result',default='log',type=str,help='use small model')
    return parser.parse_args()
args=get_args()

class Denoiser(torch.nn.Module):
    def __init__(self, model,classifier,t,flow_models=None,denoise_type='ddim',classifier_name='i3d_resnet50'):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.t = t
        self.device='cuda'
        self.diffattack=False
        self.batch_size=1
        self.flow_models=flow_models
        self.denoise_method=getattr(self,denoise_type)
        self.classifier_name=classifier_name
    

    def videopure(self,x):
        pipeline=self.model
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_vae_slicing()
        flows=self.flow_models.compute_flows(x.clone())
        inv_latents= pipeline.invert_temporal(prompt='',videos=x,inversion_t=self.t)
        output_image= pipeline.videopure(
            prompt='',
            video_latents=inv_latents,
            inversion_t=self.t,
            flows=flows,flow_model=self.flow_models
        )
        return output_image.detach()
    
    def ddim_inversion(self,x):
        pipeline=self.model
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_vae_slicing()
        inv_latents = pipeline.invert(prompt='',videos=x,inversion_t=self.t)
        output_image = pipeline.ddim_inversion(
            prompt='',
            video_latents=inv_latents,
            inversion_t=self.t,
        )
        return output_image.detach()
     
    def ddim(self,x):
        pipeline=self.model
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_vae_slicing()
        output_image = pipeline.ddim(
            prompt='',
            video_latents=x,
            inversion_t=self.t,
        )
        return output_image.detach()
        

    def ddpm(self,x):
        pipeline=self.model
        
        pipeline.enable_vae_slicing()
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        output_image = pipeline.ddpm(
            prompt='',
            video_latents=x,
            inversion_t=self.t*20,
        )
        return output_image.detach()
    
    def forward(self,x):
        
        x_pure=self.denoise_method(x)
        output=self.classifier(x_pure)
        return output,x_pure
  
    def classify(self,x):
        out=self.classifier(x)
        return out




@torch.no_grad()
def pgd_attack(x, y, net,type='normal',eps=4/255):
    loss_fn=torch.nn.CrossEntropyLoss()
    eps =eps
    alpha = 2/255
    iter =10
    no_x=x.clone()
    adv_x=x.clone().detach()
    adv_x=no_x+torch.randn_like(adv_x)*eps
    adv_x=torch.clamp(adv_x, 0, 1)
    
    for pgd_iter_id in range(iter):
            if type=='normal':
                with torch.enable_grad():
                    adv_x.requires_grad_()
                    output=net.classify(adv_x)
                    loss=loss_fn(output,y)
                    loss.backward()
                    grad=adv_x.grad.data
            elif type=='bpda':
                adv_x_pure=net.denoise_method(adv_x.clone())
                with torch.enable_grad():
                    adv_x_pure.requires_grad_()
                    output=net.classify(adv_x_pure)
                    loss=loss_fn(output,y.repeat(output.shape[0]))
                    print(loss)
                    loss.backward()
                    grad=adv_x_pure.grad.data
                    grad=grad.mean(0,keepdim=True)
            grad_sign=grad.sign()
            adv_x= adv_x + alpha*grad_sign
            delta = torch.clamp(adv_x - no_x, min=-eps, max=eps)
            adv_x=torch.clamp(no_x+delta, 0, 1) 
    x_adv=adv_x.clone().detach()
    return x_adv



def main(args,pipe):
    error=''
    test_loader=get_loaders(args.datasets,batch_size=1,n_gpus=1)
    classifier=load_classifier(args.ckpt,args.classfier).eval()
    flow_model=Flow_models(RAFT_SR())
    t=args.t
    net=Denoiser(pipe,classifier,t,flow_model,args.noise_type,args.classfier)
    net.eval()
    correct=0.
    correct_standard_acc=0.
    correct_pgd=0.
    correct_robust_acc_=0.
    correct_robust_acc=0.
    if not os.path.exists(args.result):
        os.makedirs(args.result)
    with torch.no_grad():
        with tqdm(total=len(test_loader),) as pbar:
            for i,(videos,labels) in enumerate(test_loader):
                
                videos=videos.to('cuda')
                labels=labels.to('cuda')

                #clean acc
                output=classifier(videos)
                _,pred=torch.max(output,1)
                correct+=torch.sum(pred==labels).item()

                # # Standard Acc
                output,_=net(videos.clone())
                _,pred=torch.max(output,1)
                print(pred)
                pred=torch.bincount(pred)
                pred=torch.argmax(pred)
                correct_standard_acc+=torch.sum(pred==labels).item()

                # normal attack
                if args.attack_method=='pgd':
                    x_attack=pgd_attack(videos.clone(),labels,net,'normal',eps=args.eps)
                
                # # normal attack acc without defense
                output=classifier(x_attack)
                _,pred=torch.max(output,1)
                pred=torch.bincount(pred)
                pred=torch.argmax(pred)
                correct_pgd+=torch.sum(pred==labels).item()

                # robust acc*
                output,_=net(x_attack)
                _,pred=torch.max(output,1)
                print(pred)
                pred=torch.bincount(pred)
                pred=torch.argmax(pred)
                c=torch.sum(pred==labels).item()
                if c == 0:
                    error+=str(labels.item())+' '       
                correct_robust_acc_+=c

                # adaptive attack
                if args.attack_method=='pgd':
                    x_attack=pgd_attack(videos.clone(),labels,net,'bpda',eps=args.eps)
                
                output,_=net(x_attack)
                _,pred=torch.max(output,1)
                print(pred)
                pred=torch.bincount(pred)
                pred=torch.argmax(pred)
                correct_robust_acc+=torch.sum(pred==labels).item()
                pbar.set_postfix({'Clean Acc':correct/(i+1),'Standard Acc':correct_standard_acc/(i+1), 'Normal Attack Acc':correct_pgd /(i+1), 'Robust Acc*':correct_robust_acc_/(i+1),'Robust Acc':correct_robust_acc/(i+1)})
                pbar.update(1)
                
                with open(args.result+'/{}_{}'.format(args.attack_method,args.noise_type),'w') as f:
                    f.write('num {}\n'.format(i))
                    f.write('Standard Acc:{}\n'.format(correct_standard_acc/len(test_loader)))
                    f.write('Clean Acc:{}\n'.format(correct/len(test_loader)))
                    f.write('Normal Attack Acc:{}\n'.format(correct_pgd/len(test_loader)))
                    f.write('Robust Acc*:{}\n'.format(correct_robust_acc_/len(test_loader)))
                    f.write('Robust Acc:{}\n'.format(correct_robust_acc/len(test_loader)))
                    f.write('error:{}\n'.format(error))




if __name__=='__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pipe=Video_Diffpure.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    main(args,pipe)
