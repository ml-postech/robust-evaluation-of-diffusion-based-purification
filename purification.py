import numpy as np
import torch
import torch.nn.functional as F

from utils import diff2clf, clf2diff, normalize


def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()


class PurificationForward(torch.nn.Module):
    def __init__(self, clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.max_timestep = max_timestep
        self.attack_steps = attack_steps
        self.sampling_method = sampling_method
        assert sampling_method in ['ddim', 'ddpm']
        if self.sampling_method == 'ddim':
            self.eta = 0
        elif self.sampling_method == 'ddpm':
            self.eta = 1
        self.is_imagenet = is_imagenet

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if type(t) == int:
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x

    def denoising_process(self, x, seq):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = x
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt

    def preprocess(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(noised_x, self.attack_steps[i])

        x_clf = diff2clf(x_diff)
        return x_clf

    def classify(self, x):
        logits = self.clf(x)
        return logits

    def forward(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(noised_x, self.attack_steps[i])

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        else:
            x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return logits
