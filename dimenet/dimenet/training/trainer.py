import tensorflow as tf
import tensorflow_addons as tfa
from .schedules import LinearWarmupExponentialDecay


class Trainer:
    def __init__(self, model, learning_rate=1e-3, warmup_steps=None,
                 decay_steps=100000, decay_rate=0.96,
                 ema_decay=0.999, max_grad_norm=10.0,
                 frozen_training=False,
                 early_stopping_patience=50, early_stopping_min_delta=1e-4):
        self.model = model
        self.frozen_training = frozen_training
        
        # 早停机制
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False

        if self.frozen_training:
            # 冻结训练模式已启用，参数统计将在模型构建后显示
            print("冻结训练模式已启用，参数统计将在模型构建后显示")
            self._params_counted = False
            # 初始化可训练权重列表，将在模型构建后填充
            self.trainable_weights = []


        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm

        if warmup_steps is not None:
            self.learning_rate = LinearWarmupExponentialDecay(
                learning_rate, warmup_steps, decay_steps, decay_rate)
        else:
            self.learning_rate = tf.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate)

        opt = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate, amsgrad=True)
        self.optimizer = tfa.optimizers.MovingAverage(opt, average_decay=self.ema_decay)

        # Initialize backup variables
        if model.built:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            self.backup_vars = None

    def update_weights(self, loss, gradient_tape):
        if self.frozen_training:
            # 冻结训练，只计算可训练参数的梯度
            grads = gradient_tape.gradient(loss, self.trainable_weights)
            # 过滤掉None梯度
            grads = [grad for grad in grads if grad is not None]
            trainable_weights = self.trainable_weights
        else:
            # 正常训练
            grads = gradient_tape.gradient(loss, self.model.trainable_weights)
            trainable_weights = self.model.trainable_weights

        if grads:
            global_norm = tf.linalg.global_norm(grads)
            if self.max_grad_norm is not None:
                grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm, use_norm=global_norm)

            self.optimizer.apply_gradients(zip(grads, trainable_weights))

    def load_averaged_variables(self):
        self.optimizer.assign_average_vars(self.model.trainable_weights)

    def save_variable_backups(self):
        if self.backup_vars is None:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            for var, bck in zip(self.model.trainable_weights, self.backup_vars):
                bck.assign(var)

    def restore_variable_backups(self):
        for var, bck in zip(self.model.trainable_weights, self.backup_vars):
            var.assign(bck)

    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        
        # 如果是第一次训练且模型已构建，显示参数统计
        if self.frozen_training and not hasattr(self, '_params_counted') and self.model.built:
            # 直接使用模型的可训练参数（模型已经正确设置了trainable=False）
            self.trainable_weights = [var for var in self.model.trainable_weights]
            print(f"冻结训练模式，可训练的参数数量: {len(self.trainable_weights)}")
            print(f"总参数数量: {len(self.model.weights)}")
            print(f"冻结的参数数量: {len(self.model.weights) - len(self.trainable_weights)}")
            print(f"可训练参数: {[var.name for var in self.trainable_weights]}")
            self._params_counted = True
        
        # 如果模型已构建但trainable_weights还是空的，重新设置
        if self.frozen_training and self.model.built and len(self.trainable_weights) == 0:
            self.trainable_weights = [var for var in self.model.trainable_weights]
        
        # 使用tf.function包装训练逻辑
        loss = self._train_step(inputs, targets, metrics)
        return loss
    
    def _train_step(self, inputs, targets, metrics):
        with tf.GradientTape() as tape:
            preds = self.model(inputs, training=True)
            mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
            mean_mae = tf.reduce_mean(mae)
            
            loss = mean_mae
        
        # 计算梯度
        if self.frozen_training:
            grads = tape.gradient(loss, self.trainable_weights)
        else:
            grads = tape.gradient(loss, self.model.trainable_weights)
        
        # 过滤掉None梯度
        if grads:
            grads = [grad for grad in grads if grad is not None]
            if grads:
                global_norm = tf.linalg.global_norm(grads)
                if self.max_grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm, use_norm=global_norm)
                
                # 应用梯度
                if self.frozen_training:
                    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                else:
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return loss

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)
        mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
        mean_mae = tf.reduce_mean(mae)
        loss = mean_mae

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return loss

    @tf.function
    def predict_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)

        mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
        mean_mae = tf.reduce_mean(mae)
        loss = mean_mae
        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return preds
    
    def check_early_stopping(self, current_loss):
        """检查是否应该早停"""
        if current_loss < self.best_loss - self.early_stopping_min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.early_stopping_patience:
            self.should_stop = True
            print(f"早停触发！验证损失在 {self.early_stopping_patience} 个epoch内没有改善")
            
        return self.should_stop
