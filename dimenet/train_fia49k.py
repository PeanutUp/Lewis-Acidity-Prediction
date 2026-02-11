#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import logging
import string
import random
from datetime import datetime
import yaml
import sys

# 添加当前目录到路径，以便导入 dimenet 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dimenet.model.dimenet import DimeNet
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.trainer import Trainer
from dimenet.training.metrics import Metrics
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider

def train(config):
    # 解包配置
    model_name = config['model_name']
    emb_size = config['emb_size']
    out_emb_size = config['out_emb_size']
    int_emb_size = config['int_emb_size']
    basis_emb_size = config['basis_emb_size']
    num_blocks = config['num_blocks']
    # num_bilinear 在 dimenet++ 中未使用，但在 dimenet 中使用
    num_bilinear = config.get('num_bilinear', None) 
    num_spherical = config['num_spherical']
    num_radial = config['num_radial']
    output_init = config['output_init']
    extensive = config['extensive']
    num_before_skip = config['num_before_skip']
    num_after_skip = config['num_after_skip']
    num_dense_output = config['num_dense_output']
    cutoff = config['cutoff']
    envelope_exponent = config['envelope_exponent']
    
    dataset = config['dataset']
    num_train = config['num_train']
    num_valid = config['num_valid']
    data_seed = config['data_seed']
    
    num_steps = config['num_steps']
    learning_rate = config['learning_rate']
    ema_decay = config['ema_decay']
    decay_steps = config['decay_steps']
    warmup_steps = config['warmup_steps']
    decay_rate = config['decay_rate']
    
    batch_size = config['batch_size']
    evaluation_interval = config['evaluation_interval']
    save_interval = config['save_interval']
    restart = config['restart']
    targets = config['targets']
    comment = config['comment']
    logdir = config['logdir']

    # 用于生成唯一ID
    def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    # 创建目录
    if restart is None:
        directory = (
            logdir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id_generator()
            + "_" + os.path.basename(dataset)
            + "_" + comment
        )
    else:
        directory = restart
    
    print(f"Directory: {directory}")

    print("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory)
    best_dir = os.path.join(directory, 'best')
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    log_dir = os.path.join(directory, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 保存本次训练的配置，确保复现性
    with open(os.path.join(directory, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    best_loss_file = os.path.join(best_dir, 'best_loss.npz')
    best_ckpt_file = os.path.join(best_dir, 'ckpt')
    step_ckpt_folder = log_dir

    # Initialize summary writer
    summary_writer = tf.summary.create_file_writer(log_dir)

    train = {}
    validation = {}

    # Initialize metrics
    ex = None
    
    train['metrics'] = Metrics('train', targets, ex)
    validation['metrics'] = Metrics('val', targets, ex)

    with summary_writer.as_default():
        print("Load dataset", flush=True)
        # 确保数据集路径正确
        if not os.path.isabs(dataset):
             dataset = os.path.join(os.path.dirname(__file__), dataset)

        data_container = DataContainer(dataset, cutoff=cutoff, target_keys=targets)

        # Initialize DataProvider
        data_provider = DataProvider(data_container, num_train, num_valid, batch_size,
                                     seed=data_seed, randomized=True)

        # Initialize datasets
        train['dataset'] = data_provider.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE)
        train['dataset_iter'] = iter(train['dataset'])
        validation['dataset'] = data_provider.get_dataset('val').prefetch(tf.data.experimental.AUTOTUNE)
        validation['dataset_iter'] = iter(validation['dataset'])

        print("Initialize model", flush=True)
        if model_name == "dimenet":
            model = DimeNet(
                    emb_size=emb_size, num_blocks=num_blocks, num_bilinear=num_bilinear,
                    num_spherical=num_spherical, num_radial=num_radial,
                    cutoff=cutoff, envelope_exponent=envelope_exponent,
                    num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                    num_dense_output=num_dense_output, num_targets=len(targets),
                    activation=swish, output_init=output_init)
        elif model_name == "dimenet++":
            model = DimeNetPP(
                    emb_size=emb_size, out_emb_size=out_emb_size,
                    int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
                    num_blocks=num_blocks, num_spherical=num_spherical, num_radial=num_radial,
                    cutoff=cutoff, envelope_exponent=envelope_exponent,
                    num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                    num_dense_output=num_dense_output, num_targets=len(targets),
                    activation=swish, extensive=extensive, output_init=output_init)
        else:
            raise ValueError(f"Unknown model name: '{model_name}'")

        print("Prepare training", flush=True)
        if os.path.isfile(best_loss_file):
            loss_file = np.load(best_loss_file)
            metrics_best = {k: v.item() for k, v in loss_file.items()}
        else:
            metrics_best = {k: np.inf for k in validation['metrics'].keys()}
            metrics_best['step'] = 0
            np.savez(best_loss_file, **metrics_best)

        # Initialize trainer
        trainer = Trainer(model, learning_rate, warmup_steps,
                          decay_steps, decay_rate,
                          ema_decay=ema_decay, max_grad_norm=1000)

        # Set up checkpointing
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer.optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, step_ckpt_folder, max_to_keep=3)

        # Restore latest checkpoint
        ckpt_restored = tf.train.latest_checkpoint(log_dir)
        if ckpt_restored is not None:
            ckpt.restore(ckpt_restored)

        # Training loop
        print("Start training", flush=True)
        steps_per_epoch = int(np.ceil(num_train / batch_size))

        if ckpt_restored is not None:
            step_init = ckpt.step.numpy()
        else:
            step_init = 1
        
        for step in range(step_init, num_steps + 1):
            # Update step number
            ckpt.step.assign(step)
            tf.summary.experimental.set_step(step)

            # Perform training step
            if step == step_init or step % 100 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step}: 开始 train_on_batch", flush=True)
            trainer.train_on_batch(train['dataset_iter'], train['metrics'])
            if step == step_init or step % 100 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step}: 结束 train_on_batch", flush=True)

            # Save progress
            if (step % save_interval == 0):
                manager.save()

            # Check performance on the validation set
            if (step % evaluation_interval == 0):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step}: 开始验证", flush=True)
                trainer.save_variable_backups()
                trainer.load_averaged_variables()

                for i in range(int(np.ceil(num_valid / batch_size))):
                    if i == 0:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step}: 验证第1个batch", flush=True)
                    trainer.test_on_batch(validation['dataset_iter'], validation['metrics'])
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step}: 结束验证", flush=True)

                # Update and save best result
                if validation['metrics'].mean_mae < metrics_best['mean_mae_val']:
                    metrics_best['step'] = step
                    metrics_best.update(validation['metrics'].result())

                    np.savez(best_loss_file, **metrics_best)
                    model.save_weights(best_ckpt_file)

                epoch = step // steps_per_epoch
                print(
                    f"{step}/{num_steps} (epoch {epoch+1}): "
                    f"Loss: train={train['metrics'].loss:.6f}, val={validation['metrics'].loss:.6f}; "
                    f"MAE: val={validation['metrics'].mean_mae:.6f}"
                )

                train['metrics'].write()
                validation['metrics'].write()

                train['metrics'].reset_states()
                validation['metrics'].reset_states()

                trainer.restore_variable_backups()

if __name__ == "__main__":
    # 配置 Tensorflow 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.get_logger().setLevel('WARN')
    
    # 读取配置
    config_path = os.path.join(os.path.dirname(__file__), 'config_fia49k.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)
