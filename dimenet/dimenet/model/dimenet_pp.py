import tensorflow as tf

from .layers.embedding_block import EmbeddingBlock
from .layers.bessel_basis_layer import BesselBasisLayer
from .layers.spherical_basis_layer import SphericalBasisLayer
from .layers.interaction_pp_block import InteractionPPBlock
from .layers.output_pp_block import OutputPPBlock
from .activations import swish
from .layers.minimal_output_pp_block import MinimalOutputPPBlock

class DimeNetPP(tf.keras.Model):
    """
    DimeNet++ model.

    Parameters
    ----------
    emb_size
        Embedding size used for the messages
    out_emb_size
        Embedding size used for atoms in the output block
    int_emb_size
        Embedding size used for interaction triplets
    basis_emb_size
        Embedding size used inside the basis transformation
    num_blocks
        Number of building blocks to be stacked
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    envelope_exponent
        Shape of the smooth cutoff
    cutoff
        Cutoff distance for interatomic interactions
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    extensive
        Whether the output should be extensive (proportional to the number of atoms)
    output_init
        Initialization method for the output layer (last layer in output block)
    """

    def __init__(
            self, emb_size, out_emb_size, int_emb_size, basis_emb_size,
            num_blocks, num_spherical, num_radial,
            cutoff=5.0, envelope_exponent=5, num_before_skip=1,
            num_after_skip=2, num_dense_output=3, num_targets=12,
            activation=swish, extensive=True, output_init='zeros',
            minimal_output=False, # 最小化每一个输出层
            freeze_features=False, # 冻结特征
            pretrained_weights_path=None, # 预训练权重路径
            name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.extensive = extensive

        # Cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent, name='rbf_layer')
        self.sbf_layer = SphericalBasisLayer(
            num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)

        # Embedding block
        self.emb_block = EmbeddingBlock(emb_size, activation=activation)

        # Interaction blocks
        self.int_blocks = []
        for i in range(num_blocks):
            self.int_blocks.append(
                InteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip,
                                   num_after_skip, activation=activation, name=f"int_blocks/{i}"))

        # 创建多个输出块：1个在embedding后 + num_blocks个在interaction后
        self.output_blocks = []
        
        # 第一个输出块（在embedding后）
        if minimal_output:
            self.output_blocks.append(MinimalOutputPPBlock(emb_size, num_targets,
                              activation=activation, output_init=output_init, name="output_blocks/0"))
        else:
            self.output_blocks.append(OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets,
                            activation=activation, output_init=output_init, name="output_blocks/0"))
        
        # 为每个interaction block创建输出块
        for i in range(num_blocks):
            if minimal_output:
                self.output_blocks.append(MinimalOutputPPBlock(emb_size, num_targets,
                                  activation=activation, output_init=output_init, name=f"output_blocks/{i + 1}"))
            else:
                self.output_blocks.append(OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets,
                                activation=activation, output_init=output_init, name=f"output_blocks/{i + 1}"))
        
        # 保存预训练权重路径，稍后加载
        self.pretrained_weights_path = pretrained_weights_path
        self.freeze_features = freeze_features
        
        # 冻结特征
        if freeze_features:
            self._freeze_feature_layers()

    def _freeze_feature_layers(self):
        """冻结特征层，但保持输出层可训练"""
        # 冻结基础函数层
        self.rbf_layer.trainable = False
        self.sbf_layer.trainable = False

        # 冻结嵌入层
        self.emb_block.trainable = False

        # 冻结交互层
        for block in self.int_blocks:
            block.trainable = False

        # 确保所有输出层保持可训练
        for output_block in self.output_blocks:
            output_block.trainable = True

        print(f"冻结特征层完成，输出层保持可训练")

    def load_pretrained_weights(self):
        """在模型构建后加载预训练权重"""
        if hasattr(self, 'pretrained_weights_path') and self.pretrained_weights_path:
            try:
                # 直接尝试加载权重，让TensorFlow自动处理模型构建
                self.load_weights(self.pretrained_weights_path)
                print(f"成功加载预训练权重: {self.pretrained_weights_path}")
            except Exception as e:
                print(f"加载预训练权重失败: {e}")
                print("将使用随机初始化的权重继续训练")


    def calculate_interatomic_distances(self, R, idx_i, idx_j):
        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        # ReLU prevents negative numbers in sqrt
        Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri - Rj)**2, -1)))
        return Dij

    def calculate_neighbor_angles(self, R, id3_i, id3_j, id3_k):
        """Calculate angles for neighboring atom triplets"""
        Ri = tf.gather(R, id3_i)
        Rj = tf.gather(R, id3_j)
        Rk = tf.gather(R, id3_k)
        R1 = Rj - Ri
        R2 = Rk - Rj
        x = tf.reduce_sum(R1 * R2, axis=-1)
        y = tf.linalg.cross(R1, R2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        return angle

    def call(self, inputs, training=None):
        # 在第一次调用时加载预训练权重
        if not self.built and hasattr(self, 'pretrained_weights_path') and self.pretrained_weights_path:
            # 先让模型构建完成
            pass
        
        # 如果模型已构建但特征层还没冻结，重新应用冻结
        if self.built and hasattr(self, 'freeze_features') and self.freeze_features:
            if not hasattr(self, '_features_frozen'):
                # 只冻结特征层，不重新冻结输出层
                self.rbf_layer.trainable = False
                self.sbf_layer.trainable = False
                self.emb_block.trainable = False
                for block in self.int_blocks:
                    block.trainable = False
                # 确保所有输出层保持可训练
                for output_block in self.output_blocks:
                    output_block.trainable = True
                self._features_frozen = True
        
        Z, R                         = inputs['Z'], inputs['R']
        batch_seg                    = inputs['batch_seg']
        idnb_i, idnb_j               = inputs['idnb_i'], inputs['idnb_j']
        id_expand_kj, id_reduce_ji   = inputs['id_expand_kj'], inputs['id_reduce_ji']
        id3dnb_i, id3dnb_j, id3dnb_k = inputs['id3dnb_i'], inputs['id3dnb_j'], inputs['id3dnb_k']
        n_atoms = tf.shape(Z)[0]

        # Calculate distances
        Dij = self.calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)

        # Calculate angles
        Anglesijk = self.calculate_neighbor_angles(
            R, id3dnb_i, id3dnb_j, id3dnb_k)
        sbf = self.sbf_layer([Dij, Anglesijk, id_expand_kj])

        # Embedding block
        x = self.emb_block([Z, rbf, idnb_i, idnb_j])

        # 第一个输出块（在embedding后）
        P = self.output_blocks[0]([x, rbf, idnb_i, n_atoms], training=training)

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]([x, rbf, sbf, id_expand_kj, id_reduce_ji])
            # 每个interaction block后添加输出
            P += self.output_blocks[i + 1]([x, rbf, idnb_i, n_atoms], training=training)

        if self.extensive:
            P = tf.math.segment_sum(P, batch_seg)
        else:
            P = tf.math.segment_mean(P, batch_seg)
        return P
