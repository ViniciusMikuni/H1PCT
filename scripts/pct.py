import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, Conv1D
from tensorflow.keras.models import Model
import numpy as np

    



def GetLocalFeat(pc,outsize):
    '''Return local features from embedded point cloud
    Input: point cloud shaped as (B,N,k,NFEAT)
    '''

    features = Conv2D(outsize, kernel_size=[1,1], data_format='channels_last',
                      strides=[1,1],activation='relu')(pc)
    features = Conv2D(outsize, kernel_size=[1,1], data_format='channels_last',
                      strides=[1,1],activation='relu')(features)    
    features = tf.reduce_mean(features, axis=-2)    
    return features



def GetSelfAtt(pc,mask,outsize):
    '''Get the self-attention layer
    Input: 
          Point cloud with shape (batch_size,num_point,num_dims)
          Zero-padded Mask with shape (batch_size,num_point)
    Return:
          Offset attention with shape (batch_size,num_point,outsize)
    '''
                                        
    
    query = Conv1D(outsize//4, kernel_size = 1,use_bias=False,
                   strides=1,activation=None)(pc)

    key = Conv1D(outsize//4, kernel_size = 1,use_bias=False,
                 strides=1,activation=None)(pc)
    
    key = tf.transpose(key,perm=[0,2,1]) #B,C//4,N

    value = Conv1D(outsize, kernel_size = 1,use_bias=False,
                   strides=1,activation=None)(pc)    
    
    value = tf.transpose(value,perm=[0,2,1]) #B,C,N

    energy = tf.matmul(query,key) #B,N,N    

    #Make zero-padded less important

    mask_offset = -1000*mask+tf.ones_like(mask)
    mask_matrix = tf.matmul(tf.expand_dims(mask_offset,-1),tf.transpose(tf.expand_dims(mask_offset,-1),perm=[0,2,1]))
    mask_matrix = mask_matrix - tf.ones_like(mask_matrix)
    energy = energy + mask_matrix
    attention = tf.keras.activations.softmax(energy)
    zero_mask = tf.where(tf.equal(mask_matrix,0),tf.ones_like(mask_matrix),tf.zeros_like(mask_matrix))  

    attention = attention*zero_mask

    attention = attention / (1e-9 + tf.reduce_sum(attention,1, keepdims=True))
    self_att = tf.matmul(value,attention) #B,C,N
    self_att = tf.transpose(self_att,perm=[0,2,1]) #B,N,C
    

    self_att = Conv1D(outsize, kernel_size = 1,use_bias=False,
                   strides=1,activation=None)(pc-self_att)    
    
    return pc+self_att,attention


def GetEdgeFeat(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
    point_cloud: (batch_size, num_points, 1, num_dims) 
    nn_idx: (batch_size, num_points, k)
    k: int
    Returns:
    edge features: (batch_size, num_points, k, num_dims)
    """



    point_cloud_central = point_cloud

    batch_size = tf.shape(point_cloud)[0]
    num_points = tf.shape(point_cloud)[1]
    num_dims = point_cloud.get_shape()[2]



    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)


    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
    return edge_feature

def pairwise_distanceR(point_cloud, mask=None):
    """Compute pairwise distance in the eta-phi plane for the point cloud.
    Uses the third dimension to find the zero-padded terms
    Args:
      point_cloud: tensor (batch_size, num_points, 2)
      IMPORTANT: The order should be (eta, phi)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    point_cloud = point_cloud[:,:,:2] #eta-phi
  
    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_phi = point_cloud_transpose[:,1:,:]
    point_cloud_phi = tf.tile(point_cloud_phi,[1,point_cloud_phi.get_shape()[2],1])
    point_cloud_phi_transpose = tf.transpose(point_cloud_phi,perm=[0, 2, 1])
    point_cloud_phi = tf.abs(point_cloud_phi - point_cloud_phi_transpose)
    is_bigger2pi = tf.greater_equal(tf.abs(point_cloud_phi),2*np.pi)
    point_cloud_phi_corr = tf.where(is_bigger2pi,4*np.pi**2-4*np.pi*point_cloud_phi,point_cloud_phi-point_cloud_phi)
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose) # x.x + y.y + z.z shape: NxN
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True) # from x.x, y.y, z.z to x.x + y.y + z.z
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])

    if mask != None:
        zero_mask = 10000*tf.expand_dims(mask,-1)
        zero_mask_transpose = tf.transpose(zero_mask, perm=[0, 2, 1])
        zero_mask = zero_mask + zero_mask_transpose
        zero_mask = tf.where(tf.equal(zero_mask,20000),tf.zeros_like(zero_mask),zero_mask)

    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose +point_cloud_phi_corr+ zero_mask,zero_mask

def pairwise_distance(point_cloud, mask=None): 
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1]) 
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose) # x.x + y.y + z.z shape: NxN
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True) # from x.x, y.y, z.z to x.x + y.y + z.z
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    # is_zero = point_cloud[:,:,] 
    # point_shift = 1000*tf.where(tf.equal(pt,0),tf.ones_like(pt),tf.fill(tf.shape(pt), tf.constant(0.0, dtype=pt.dtype)))
    if mask != None:
        return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose + mask
    else:
        return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int

    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    _, nn_idx = tf.math.top_k(neg_adj, k=k)  # values, indices
    return nn_idx


def PCT(npoints,nvars):
    inputs = Input((npoints,nvars))
    input_q2 = Input((1,))

    batch_size = tf.shape(inputs)[0]
            
    k = 5
    mask = tf.where(inputs[:,:,2]==0,K.ones_like(inputs[:,:,2]),K.zeros_like(inputs[:,:,2]))
    adj,mask_matrix = pairwise_distanceR(inputs[:,:,:3],mask)    
    nn_idx = knn(adj, k=k)    

    edge_feature_0 = GetEdgeFeat(inputs, nn_idx=nn_idx, k=k)    
    features_0 = GetLocalFeat(edge_feature_0,64)

    # adj = pairwise_distance(features_0,mask_matrix)
    # nn_idx = knn(adj, k=k)

    # edge_feature_1 = GetEdgeFeat(features_0, nn_idx=nn_idx, k=k)    
    # features_1 = GetLocalFeat(edge_feature_1,32)
    
    self_att_1,attention1 = GetSelfAtt(features_0,mask,64)
    self_att_2,attention2 = GetSelfAtt(self_att_1,mask,64)
    #self_att_3,attention3 = GetSelfAtt(self_att_2,mask,64)

    concat = tf.concat([
        self_att_1,
        self_att_2,
        #self_att_3,    
        features_0,
     ]
        ,axis=-1)


    net = Conv1D(128, kernel_size = 1,
                 strides=1,activation='relu',
             )(concat)

    net = tf.reduce_mean(net, axis=1)

    net = tf.concat([net,input_q2],axis=-1)
    
    net = Dense(64,activation='relu')(net)
    net = Dropout(0.5)(net)
    outputs = Dense(1,activation='sigmoid')(net)
    
    return inputs,input_q2,outputs
