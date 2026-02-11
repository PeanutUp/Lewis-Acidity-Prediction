import tensorflow as tf
from tensorflow.python.keras import layers
from ..initializers import GlorotOrthogonal

class MinimalOutputPPBlock(layers.Layer):
    def __init__(self, emb_size, num_targets=1, activation=None, 
                 output_init='zeros', name='minimal_output', **kwargs):
        super().__init__(name=name, **kwargs)
        weight_init = GlorotOrthogonal()
        if output_init == 'GlorotOrthogonal':
            output_init = GlorotOrthogonal()
        elif output_init == 'zeros':
            output_init = 'zeros'
        
        # 只保留RBF处理
        self.dense_rbf = layers.Dense(emb_size, use_bias=False,
                          kernel_initializer=weight_init, name="dense_rbf")
        
        # 添加up_projection来匹配预训练权重，但跳过中间的dense层
        self.up_projection = layers.Dense(256, use_bias=False,
                         kernel_initializer=weight_init, name="up_projection")
        
        
        # 直接输出，无中间层
        self.dense_final = layers.Dense(num_targets, use_bias=False,
                        kernel_initializer=output_init, name="dense_final")
    
    def call(self, inputs, training=None):
        x, rbf, idnb_i, n_atoms = inputs
        
        # RBF门控机制
        g = self.dense_rbf(rbf)
        x = g * x
        
        # 聚合到原子级别
        x = tf.math.unsorted_segment_sum(x, idnb_i, n_atoms)
        
        # 使用up_projection来匹配预训练权重的维度
        x = self.up_projection(x)
        
        # 直接输出，无中间层
        x = self.dense_final(x)
        return x