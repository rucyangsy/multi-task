import copy
import glob
import os
import time
import pprint
from collections import deque
from collections import OrderedDict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.function import AverageMeter
from evaluation import evaluate

from tensorboardX import SummaryWriter
from utils.utils import create_logger
from utils.utils import save_checkpoint
from a2c_ppo_acktr.config import config
from a2c_ppo_acktr.config import update_config
from a2c_ppo_acktr.config import update_dir
from a2c_ppo_acktr.config import get_model_name


def main():
    args = get_args()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.cuda and torch.cuda.is_available() and config.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train', seed=config.seed)

    eval_log_dir = final_output_dir + "_eval"

    utils.cleanup_log_dir(final_output_dir)
    utils.cleanup_log_dir(eval_log_dir)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer = SummaryWriter(tb_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:" + config.GPUS if config.cuda else "cpu")

    width = height = 84

    envs = []
    for env_name in config.env_name:
        env = make_vec_envs(config.env_name, config.seed, config.num_processes,
                             config.gamma, final_output_dir, device, False,
                             width=width, height=height, ram_wrapper=False)
        envs.append(env)
    # create agent
    actor_critic = Policy(
        envs[0].observation_space.shape,
        base_kwargs={'recurrent': config.recurrent_policy,
                     'hidden_size': config.hidden_size,
                     'feat_from_selfsup_attention': config.feat_from_selfsup_attention,
                     'feat_add_selfsup_attention': config.feat_add_selfsup_attention,
                     'feat_mul_selfsup_attention_mask': config.feat_mul_selfsup_attention_mask,
                     'selfsup_attention_num_keypoints': config.SELFSUP_ATTENTION.NUM_KEYPOINTS,
                     'selfsup_attention_gauss_std':  config.SELFSUP_ATTENTION.GAUSS_STD,
                     'selfsup_attention_fix': config.selfsup_attention_fix,
                     'selfsup_attention_fix_keypointer': config.selfsup_attention_fix_keypointer,
                     'selfsup_attention_pretrain': config.selfsup_attention_pretrain,
                     'selfsup_attention_keyp_maps_pool': config.selfsup_attention_keyp_maps_pool,
                     'selfsup_attention_image_feat_only': config.selfsup_attention_image_feat_only,
                     'selfsup_attention_feat_masked': config.selfsup_attention_feat_masked,
                     'selfsup_attention_feat_masked_residual': config.selfsup_attention_feat_masked_residual,
                     'selfsup_attention_feat_load_pretrained': config.selfsup_attention_feat_load_pretrained,
                     'use_layer_norm': config.use_layer_norm,
                     'selfsup_attention_keyp_cls_agnostic': config.SELFSUP_ATTENTION.KEYPOINTER_CLS_AGNOSTIC,
                     'selfsup_attention_feat_use_ln': config.SELFSUP_ATTENTION.USE_LAYER_NORM,
                     'selfsup_attention_use_instance_norm': config.SELFSUP_ATTENTION.USE_INSTANCE_NORM,
                     'feat_mul_selfsup_attention_mask_residual': config.feat_mul_selfsup_attention_mask_residual,
                     'bottom_up_form_objects': config.bottom_up_form_objects,
                     'bottom_up_form_num_of_objects': config.bottom_up_form_num_of_objects,
                     'gaussian_std': config.gaussian_std,
                     'train_selfsup_attention': config.train_selfsup_attention,
                     'block_selfsup_attention_grad': config.block_selfsup_attention_grad,
                     'sep_bg_fg_feat': config.sep_bg_fg_feat,
                     'mask_threshold': config.mask_threshold,
                     'fix_feature': config.fix_feature
                     })

    # init / load parameter
    if config.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.MODEL_FILE))
        state_dict = torch.load(config.MODEL_FILE)

        state_dict = OrderedDict((_k, _v) for _k, _v in state_dict.items() if 'dist' not in _k)

        actor_critic.load_state_dict(state_dict, strict=False)
    elif config.RESUME:
        checkpoint_file = os.path.join(
            final_output_dir, 'checkpoint.pth'
        )
        if os.path.exists(checkpoint_file):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            actor_critic.load_state_dict(checkpoint['state_dict'])

            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))

    actor_critic.to(device)

    if config.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            alpha=config.alpha,
            max_grad_norm=config.max_grad_norm,
            train_selfsup_attention=config.train_selfsup_attention)
    elif config.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            config.clip_param,
            config.ppo_epoch,
            config.num_mini_batch,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm)
    elif config.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, config.value_loss_coef, config.entropy_coef, acktr=True,
            train_selfsup_attention=config.train_selfsup_attention,
            max_grad_norm=config.max_grad_norm
        )

    # rollouts: environment
    rollouts = RolloutStorage(config.num_steps, config.num_processes,
                              envs[0].observation_space.shape,
                              keep_buffer=config.train_selfsup_attention,
                              buffer_size=config.train_selfsup_attention_buffer_size)

    if config.RESUME:
        if os.path.exists(checkpoint_file):
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
    
    obs = []
    for i, env in enumerate(envs):
        obs.append(env.reset())
        rollouts.obs[i].copy_(obs[-1])
    #obs = envs.reset()
    
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        config.num_env_steps) // config.num_steps // config.num_processes
    best_perf = 0.0
    best_model = False
    print('num updates', num_updates, 'num steps', config.num_steps)

    for j in range(num_updates):

        if config.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if config.algo == "acktr" else config.lr)

        for step in range(config.num_steps):
            
            # Sample actions
            for num in config.env_num: 
                action = env[num].action_space.sample()
                # Obser reward and next obs
                obs, reward, done, infos = envs[num].step(action)

                rollouts.insert(obs)

        if config.train_selfsup_attention and j > 15:
            for _iter in range(config.num_steps // 5):
                frame_x, frame_y = rollouts.generate_pair_image()
                selfsup_attention_loss, selfsup_attention_output, image_b_keypoints_maps = \
                    agent.update_selfsup_attention(frame_x, frame_y, config.SELFSUP_ATTENTION)

        if j % config.log_interval == 0 and len(episode_rewards) > 1:
            if config.train_selfsup_attention and j > 15:
                msg = msg + 'selfsup attention loss {:.5f}\n'.format(selfsup_attention_loss)
            logger.info(msg)

        if (config.eval_interval is not None and len(episode_rewards) > 1
                and j % config.eval_interval == 0):
            total_num_steps = (j + 1) * config.num_processes * config.num_steps
            ob_rms = getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            eval_mean_score, eval_max_score, eval_scores = evaluate(actor_critic, ob_rms, config.env_name, config.seed,
                                                                    config.num_processes, eval_log_dir, device,
                                                                    width=width, height=height)
            perf_indicator = eval_mean_score
            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            # record test scores
            with open(os.path.join(final_output_dir, 'test_scores'), 'a+') as f:
                out_s = "TEST: {}, {}, {}, {}\n".format(str(total_num_steps), str(eval_mean_score), str(eval_max_score),
                                                [str(_eval_scores) for _eval_scores in eval_scores])
                print(out_s, end="", file=f)
                logger.info(out_s)
            writer.add_scalar('data/mean_score', eval_mean_score, total_num_steps)
            writer.add_scalar('data/max_score', eval_max_score, total_num_steps)

            writer.add_scalars('test', {'mean_score': eval_mean_score}, total_num_steps)

            # save for every interval-th episode or for the last epoch
            if (j % config.save_interval == 0
                or j == num_updates - 1) and config.save_dir != "":

                logger.info("=> saving checkpoint to {}".format(final_output_dir))
                epoch = j / config.save_interval
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': get_model_name(config),
                    'state_dict': actor_critic.state_dict(),
                    'perf': perf_indicator,
                    'optimizer': agent.optimizer.state_dict(),
                    'ob_rms': getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(actor_critic.state_dict(), final_model_state_file)

    # export_scalars_to_json needs results from add scalars
    writer.export_scalars_to_json(os.path.join(tb_log_dir, 'all_scalars.json'))
    writer.close()


if __name__ == "__main__":
    main()
