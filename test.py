import os
import datetime
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision.utils as tvu
import wandb

from attacks.pgd_eot import PGD
from attacks.pgd_eot_l2 import PGDL2
from attacks.pgd_eot_bpda import BPDA
from load_data import load_dataset_by_name
from load_model import load_models
from purification import PurificationForward
from utils import copy_source
from path import *


def get_diffusion_params(max_timesteps, num_denoising_steps):
    max_timestep_list = [int(i) for i in max_timesteps.split(',')]
    num_denoising_steps_list = [int(i) for i in num_denoising_steps.split(',')]
    assert len(max_timestep_list) == len(num_denoising_steps_list)

    diffusion_steps = []
    for i in range(len(max_timestep_list)):
        diffusion_steps.append([i - 1 for i in range(max_timestep_list[i] // num_denoising_steps_list[i],
                               max_timestep_list[i] + 1, max_timestep_list[i] // num_denoising_steps_list[i])])
        max_timestep_list[i] = max_timestep_list[i] - 1

    return max_timestep_list, diffusion_steps


def predict(x, args, defense_forward, num_classes):
    ensemble = torch.zeros(x.shape[0], num_classes).to(x.device)
    for _ in range(args.num_ensemble_runs):
        _x = x.clone()

        logits = defense_forward(_x)
        pred = logits.max(1, keepdim=True)[1]
        
        for idx in range(x.shape[0]):
            ensemble[idx, pred[idx]] += 1

    pred = ensemble.max(1, keepdim=True)[1]
    return pred


def test(rank, gpu, args):
    print('rank {} | gpu {} started'.format(rank, gpu))
    if rank == 0 and args.use_wandb:
        wandb.init(project=args.wandb_project_name)

    model_src = diffusion_model_path[args.dataset]
    is_imagenet = True if args.dataset == 'imagenet' else False
    dataset_root = imagenet_path if is_imagenet else './dataset'
    num_classes = 1000 if is_imagenet else 10

    # Set test directory name
    exp_dir = './result/{}/{}-max_t_{}-denoising_step-{}-att_{}'.format(
        args.exp,
        args.def_sampling_method,
        args.def_max_timesteps,
        args.def_num_denoising_steps,
        args.att_num_denoising_steps,
    )
    if rank == 0:
        os.makedirs('./result', exist_ok=True)
        os.makedirs('./result/{}'.format(args.exp), exist_ok=True)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs('{}/imgs'.format(exp_dir), exist_ok=True)
        copy_source(__file__, exp_dir)

    # Device
    device = torch.device('cuda:{}'.format(gpu))

    # Load dataset
    assert 512 % args.batch_size == 0
    testset = load_dataset_by_name(args.dataset, dataset_root, 512)
    testsampler = torch.utils.data.distributed.DistributedSampler(testset,
                                                                  num_replicas=args.world_size,
                                                                  rank=rank)
    testLoader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=testsampler,
                                             drop_last=False)

    # Load models
    clf, diffusion = load_models(args, model_src, device)

    # Process diffusion hyperparameters
    def_max_timesteps, def_diffusion_steps = get_diffusion_params(
        args.def_max_timesteps, args.def_num_denoising_steps)
    att_max_timesteps, att_diffusion_steps = get_diffusion_params(
        args.att_max_timesteps, args.att_num_denoising_steps)

    print('def_max_timesteps: ', def_max_timesteps)
    print('def_diffusion_steps: ', def_diffusion_steps)
    print('def_sampling_method: ', args.def_sampling_method)

    print('att_max_timesteps: ', att_max_timesteps)
    print('att_diffusion_steps: ', att_diffusion_steps)
    print('att_sampling_method: ', args.att_sampling_method)

    # Set diffusion process for attack and defense
    attack_forward = PurificationForward(
        clf, diffusion, att_max_timesteps, att_diffusion_steps, args.att_sampling_method, is_imagenet, device)
    defense_forward = PurificationForward(
        clf, diffusion, def_max_timesteps, def_diffusion_steps, args.def_sampling_method, is_imagenet, device)

    # Set adversarial attack
    if args.dataset == 'cifar10':
        print('[Dataset] CIFAR-10')
        if args.attack_method == 'pgd':  # PGD Linf
            eps = 8./255.
            attack = PGD(attack_forward, attack_steps=args.n_iter,
                         eps=eps, step_size=0.007, eot=args.eot)
            print('[Attack] PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))
        elif args.attack_method == 'pgd_l2':  # PGD L2
            eps = 0.5
            attack = PGDL2(attack_forward, attack_steps=args.n_iter,
                           eps=eps, step_size=0.007, eot=args.eot)
            print('[Attack] PGD L2 | attack_steps: {} | eps: {} | eot: {}'.format(
                args.n_iter, eps, args.eot))
        elif args.attack_method == 'bpda':  # BPDA
            eps = 8./255.
            attack = BPDA(attack_forward, attack_steps=args.n_iter,
                          eps=eps, step_size=0.007, eot=args.eot)
            print('[Attack] BPDA Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))
    elif args.dataset == 'imagenet':
        print('[Dataset] ImageNet')
        eps = 4./255.
        attack = PGD(attack_forward, attack_steps=args.n_iter,
                     eps=eps, step_size=0.007, eot=args.eot)
        print('[Attack] ImageNet | PGD Linf | attack_steps: {} | eps: {} | eot: {}'.format(
            args.n_iter, eps, args.eot))
    elif args.dataset == 'svhn':
        print('[Dataset] SVHN')
        eps = 8./255.
        attack = PGD(attack_forward, attack_steps=args.n_iter,
                     eps=eps, step_size=0.007, eot=args.eot)
        print('[Attack] PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
            args.n_iter, eps, args.eot))

    dist.barrier()

    correct_nat = torch.tensor([0]).to(device)
    correct_adv = torch.tensor([0]).to(device)
    total = torch.tensor([0]).to(device)
    for idx, (x, y) in enumerate(testLoader):
        x = x.to(device)
        y = y.to(device)

        clf.eval()
        diffusion.eval()

        x_adv = attack(x, y)

        if idx < 10 and rank == 0:
            tvu.save_image(x, '{}/imgs/{}_x.png'.format(exp_dir, idx))
            tvu.save_image(x_adv, '{}/imgs/{}_x_adv.png'.format(exp_dir, idx))

        with torch.no_grad():
            pred_nat = predict(x, args, defense_forward, num_classes)
            correct_nat += pred_nat.eq(y.view_as(pred_nat)).sum().item()

            pred_adv = predict(x_adv, args, defense_forward, num_classes)
            correct_adv += pred_adv.eq(y.view_as(pred_adv)).sum().item()

        total += x.shape[0]

        print('rank {} | {} | num_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
            rank, idx, total.item(), (correct_nat / total *
                                      100).item(), (correct_adv / total * 100).item()
        ))
        if rank == 0 and args.use_wandb:
            wandb.log({
                'acc_nat': correct_nat / total * 100,
                'acc_adv': correct_adv / total * 100
            })

    dist.barrier()

    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_nat, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_adv, op=dist.ReduceOp.SUM)
    print('rank {} | num_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
        rank, total.item(), (correct_nat / total *
                             100).item(), (correct_adv / total * 100).item()
    ))
    if rank == 0 and args.use_wandb:
        wandb.log({
            'total_acc_nat': correct_nat / total * 100,
            'total_acc_adv': correct_adv / total * 100
        })


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument("--use_cuda", action='store_true',
                        help="Whether use gpu or not")
    parser.add_argument("--use_wandb", action='store_true',
                        help="Whether use wandb or not")
    parser.add_argument("--wandb_project_name",
                        default='test', help="Wandb project name")
    parser.add_argument('--exp', type=str, required=True,
                        default='test', help='Experiment name')
    parser.add_argument("--dataset", type=str, default='cifar10',
                        choices=['cifar10', 'imagenet', 'svhn'])
    parser.add_argument('--batch_size', type=int, default=128)

    # Attack
    parser.add_argument("--attack_method", type=str, default='pgd',
                        choices=['pgd', 'pgd_l2', 'bpda'], required=True)
    parser.add_argument('--n_iter', type=int, default=200,
                        help='The nubmer of iterations for the attack generation')
    parser.add_argument('--eot', type=int, default=20,
                        help='The number of EOT samples for the attack')

    # Purification hyperparameters in defense
    parser.add_argument("--def_max_timesteps", type=str, required=True,
                        help='The number of forward steps for each purification step in defense')
    parser.add_argument('--def_num_denoising_steps', type=str, required=True,
                        help='The number of denoising steps for each purification step in defense')
    parser.add_argument('--def_sampling_method', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                        help='Sampling method for the purification in defense')
    parser.add_argument('--num_ensemble_runs', type=int, default=1,
                        help='The number of ensemble runs for purification in defense')

    # Purification hyperparameters in attack generation
    parser.add_argument("--att_max_timesteps", type=str, required=True,
                        help='The number of forward steps for each purification step in attack')
    parser.add_argument('--att_num_denoising_steps', type=str, required=True,
                        help='The number of denoising steps for each purification step in attack')
    parser.add_argument('--att_sampling_method', type=str,
                        help='Sampling method for the purification in attack')

    # Torch DDP
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='Number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost',
                        help='Address for master')
    parser.add_argument('--port', type=str, default='1234',
                        help='Port number for torch ddp')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args


def cleanup():
    dist.destroy_process_group()


def init_processes(rank, size, fn, args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size,
                            timeout=datetime.timedelta(hours=4))
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    print(args)

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            p = Process(target=init_processes, args=(
                global_rank, global_size, test, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        init_processes(0, size, test, args)
