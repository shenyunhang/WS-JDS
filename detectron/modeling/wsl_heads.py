from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
from detectron.modeling.ResNet import add_stage

from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

# ---------------------------------------------------------------------------- #
# WSL outputs and losses
# ---------------------------------------------------------------------------- #


def add_wsl_outputs(model, blob_in, dim, prefix=''):
    """Add RoI classification and bounding box regression output ops."""
    if cfg.WSL.CONTEXT:
        return add_wsl_context_outputs(model, blob_in, dim)
    # Box classification layer
    model.FC(
        blob_in,
        prefix + 'fc8c',
        dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))
    model.FC(
        blob_in,
        prefix + 'fc8d',
        dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))

    model.Softmax(prefix + 'fc8c', prefix + 'alpha_cls', axis=1)
    model.Transpose(prefix + 'fc8d', prefix + 'fc8d_t', axes=(1, 0))
    model.Softmax(prefix + 'fc8d_t', prefix + 'alpha_det_t', axis=1)
    model.Transpose(prefix + 'alpha_det_t', prefix + 'alpha_det', axes=(1, 0))
    model.net.Mul([prefix + 'alpha_cls', prefix + 'alpha_det'],
                  prefix + 'rois_pred')

    if not model.train:  # == if test
        # Add BackGround predictions
        model.net.Split(
            prefix + 'rois_pred', [prefix + 'rois_bg_pred', prefix + 'notuse'],
            split=[1, model.num_classes - 2],
            axis=1)
        model.net.Concat(
            [prefix + 'rois_bg_pred', prefix + 'rois_pred'],
            [prefix + 'cls_prob', prefix + 'cls_prob_concat_dims'],
            axis=1)


def add_wsl_context_outputs(model, blobs_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    model.FC(
        blobs_in[0],
        'fc8c',
        dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))
    model.FC(
        blobs_in[1],
        'fc8d_frame',
        dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))
    model.net.FC([blobs_in[2], 'fc8d_frame_w', 'fc8d_frame_b'], 'fc8d_context')
    model.net.Sub(['fc8d_frame', 'fc8d_context'], 'fc8d')
    model.Softmax('fc8c', 'alpha_cls', axis=1)
    model.Transpose('fc8d', 'fc8d_t', axes=(1, 0))
    model.Softmax('fc8d_t', 'alpha_det_t', axis=1)
    model.Transpose('alpha_det_t', 'alpha_det', axes=(1, 0))
    model.net.Mul(['alpha_cls', 'alpha_det'], 'rois_pred')

    if not model.train:  # == if test
        # model.net.Alias('rois_pred', 'cls_prob')
        # Add BackGround predictions
        model.net.Split(
            'rois_pred', ['rois_bg_pred', 'notuse'],
            split=[1, model.num_classes - 2],
            axis=1)
        model.net.Concat(['rois_bg_pred', 'rois_pred'],
                         ['cls_prob', 'cls_prob_concat_dims'],
                         axis=1)


def add_cls_pred(in_blob, out_blob, model, prefix=''):
    assert cfg.TRAIN.IMS_PER_BATCH == 1, 'Only support one image per GPU'

    if False:
        model.net.RoIScoreReshape([in_blob, 'rois'],
                                  in_blob + '_reshape',
                                  num_classes=model.num_classes - 1,
                                  batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                  rois_size=cfg.TRAIN.BATCH_SIZE_PER_IM)
        model.net.RoIScorePool(
            in_blob + '_reshape', out_blob, num_classes=model.num_classes - 1)

        return

    model.net.ReduceSum(in_blob, out_blob, axes=[0], keepdims=True)


def add_center_loss(label_blob, pred_blob, feature_blob, feature_dims, model):
    CF = model.create_param(
        param_name='center_feature',
        initializer=initializers.Initializer("GaussianFill"),
        tags=ParameterTags.COMPUTED_PARAM,
        shape=[
            model.num_classes - 1, cfg.WSL.CENTER_LOSS_NUMBER, feature_dims
        ],
    )

    dCF = model.create_param(
        param_name='center_feature_g',
        initializer=initializers.Initializer("ConstantFill", value=0.0),
        # tags=ParameterTags.COMPUTED_PARAM,
        shape=[
            model.num_classes - 1, cfg.WSL.CENTER_LOSS_NUMBER, feature_dims
        ],
    )

    ndCF = model.create_param(
        param_name='center_feature_n_u',
        initializer=initializers.Initializer("ConstantFill", value=0.0),
        # tags=ParameterTags.COMPUTED_PARAM,
        shape=[model.num_classes - 1, cfg.WSL.CENTER_LOSS_NUMBER],
    )

    if cfg.WSL.CPG or cfg.WSL.CSC:
        input_blobs = [
            label_blob, pred_blob, feature_blob, CF, dCF, ndCF, 'cpg'
        ]
    else:
        input_blobs = [label_blob, pred_blob, feature_blob, CF, dCF, ndCF]

    output_blobs = ['loss_center', 'D', 'S']

    loss_center, D, S = model.net.CenterLoss(
        input_blobs,
        output_blobs,
        max_iter=cfg.WSL.CSC_MAX_ITER,
        top_k=cfg.WSL.CENTER_LOSS_TOP_K,
        display=int(1280 / cfg.NUM_GPUS),
        update=int(128 / cfg.NUM_GPUS))

    loss_gradients = get_loss_gradients_weighted(model, [loss_center], 0.4096)
    model.AddLosses(['loss_center'])

    return loss_gradients


def add_min_entropy_loss(model, pred, label, loss, cpg=None):
    in_blobs = [pred, label]
    if cpg:
        in_blobs.append(cpg)
    out_blobs = [loss]
    loss_entropy = model.net.MinEntropyLoss(in_blobs, out_blobs)

    loss_gradients = get_loss_gradients_weighted(model, [loss_entropy], 0.1)
    model.AddLosses([loss])

    return loss_gradients


def add_cross_entropy_loss(model, pred, label, loss, weight=None, cpg=None):
    in_blob = [pred, label]
    if cpg:
        in_blob.append(cpg)
    out_blob = [loss]

    if weight:
        in_blob.insert(2, weight)
        model.net.WeightedCrossEntropyWithLogits(in_blob, out_blob)
    else:
        model.net.CrossEntropyWithLogits(in_blob, out_blob)


def add_csc_loss(model,
                 cpg_blob='cpg',
                 cls_prob_blob='cls_prob',
                 rois_pred_blob='rois_pred',
                 rois_blob='rois',
                 loss_weight=1.0,
                 csc_layer='CSC',
                 prefix='',
                 **kwargs):
    csc_func = getattr(model.net, csc_layer)
    csc_args = {}
    csc_args['tau'] = cfg.WSL.CPG_TAU
    csc_args['max_iter'] = cfg.WSL.CSC_MAX_ITER
    # csc_args['debug_info'] = cfg.WSL.DEBUG
    csc_args['fg_threshold'] = cfg.WSL.CSC_FG_THRESHOLD
    csc_args['mass_threshold'] = cfg.WSL.CSC_MASS_THRESHOLD
    csc_args['density_threshold'] = cfg.WSL.CSC_DENSITY_THRESHOLD
    csc_args.update(kwargs)
    csc, labels_oh_pos, labels_oh_neg = csc_func(
        [cpg_blob, 'labels_oh', cls_prob_blob, rois_blob],
        [prefix + 'csc', prefix + 'labels_oh_pos', prefix + 'labels_oh_neg'],
        **csc_args)

    model.net.CSCConstraint([rois_pred_blob, csc],
                            [prefix + 'rois_pred_pos', prefix + 'csc_pos'],
                            polar=True)
    model.net.CSCConstraint([rois_pred_blob, csc],
                            [prefix + 'rois_pred_neg', prefix + 'csc_neg'],
                            polar=False)

    add_cls_pred(prefix + 'rois_pred_pos', prefix + 'cls_prob_pos', model)
    add_cls_pred(prefix + 'rois_pred_neg', prefix + 'cls_prob_neg', model)

    weight = None

    add_cross_entropy_loss(
        model,
        prefix + 'cls_prob_pos',
        prefix + 'labels_oh_pos',
        prefix + 'cross_entropy_pos',
        cpg=cpg_blob,
        weight=weight)

    add_cross_entropy_loss(
        model,
        prefix + 'cls_prob_neg',
        prefix + 'labels_oh_neg',
        prefix + 'cross_entropy_neg',
        cpg=cpg_blob,
        weight=weight)

    loss_cls_pos = model.net.AveragedLoss([prefix + 'cross_entropy_pos'],
                                          [prefix + 'loss_cls_pos'])
    loss_cls_neg = model.net.AveragedLoss([prefix + 'cross_entropy_neg'],
                                          [prefix + 'loss_cls_neg'])

    # loss_gradients = blob_utils.get_loss_gradients(
    # model, [loss_cls_pos, loss_cls_neg])
    loss_gradients = get_loss_gradients_weighted(
        model, [loss_cls_pos, loss_cls_neg], loss_weight)
    model.Accuracy([prefix + 'cls_prob_pos', 'labels_int32'],
                   prefix + 'accuracy_cls_pos')
    # model.Accuracy(['cls_prob_neg', 'labels_int32'], 'accuracy_cls_neg')
    model.AddLosses([prefix + 'loss_cls_pos', prefix + 'loss_cls_neg'])
    # model.AddMetrics(['accuracy_cls_pos', 'accuracy_cls_neg'])
    model.AddMetrics([prefix + 'accuracy_cls_pos'])

    return loss_gradients


def add_wsl_losses(model, prefix=''):
    add_cls_pred(prefix + 'rois_pred', prefix + 'cls_prob', model, prefix='')
    classes_weight = None

    cpg = None
    if cfg.WSL.CPG or cfg.WSL.CSC:
        cpg_args = {}
        cpg_args['tau'] = cfg.WSL.CPG_TAU
        cpg_args['max_iter'] = max(cfg.WSL.CPG_MAX_ITER, cfg.WSL.CSC_MAX_ITER)
        # cpg_args['debug_info'] = cfg.WSL.DEBUG
        cpg_args['cpg_net_name'] = model.net.Proto().name + '_cpg'
        cpg_args['pred_blob_name'] = cfg.WSL.CPG_PRE_BLOB
        cpg_args['data_blob_name'] = cfg.WSL.CPG_DATA_BLOB

        model.net.CPG(['labels_oh', prefix + 'cls_prob'], ['cpg_raw'],
                      **cpg_args)
        model.net.CPGScale(['cpg_raw', 'labels_oh', prefix + 'cls_prob'],
                           'cpg',
                           tau=cfg.WSL.CPG_TAU)
        cpg = 'cpg'

    if cfg.WSL.CSC:
        if not cfg.MODEL.MASK_ON or True:
            loss_gradients = add_csc_loss(
                model,
                'cpg',
                prefix + 'cls_prob',
                prefix + 'rois_pred',
                prefix + 'rois',
                loss_weight=1.0,
                prefix='')
        else:
            loss_gradients = {}
    else:
        add_cross_entropy_loss(
            model,
            prefix + 'cls_prob',
            'labels_oh',
            prefix + 'cross_entropy',
            weight=classes_weight,
            cpg=cpg)
        loss_cls = model.net.AveragedLoss([prefix + 'cross_entropy'],
                                          [prefix + 'loss_cls'])

        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls])
        model.Accuracy([prefix + 'cls_prob', 'labels_int32'],
                       prefix + 'accuracy_cls')
        model.AddLosses([prefix + 'loss_cls'])
        model.AddMetrics(prefix + 'accuracy_cls')

    if cfg.WSL.CENTER_LOSS:
        center_dim = 4096
        rois_pred = prefix + 'rois_pred'

        loss_gradients_center = add_center_loss(
            'labels_oh', rois_pred, prefix + 'drop7', center_dim, model)
        loss_gradients.update(loss_gradients_center)

    if cfg.WSL.MIN_ENTROPY_LOSS:
        loss_gradients_ME = add_min_entropy_loss(
            model,
            prefix + 'rois_pred',
            'labels_oh',
            prefix + 'loss_entropy',
            cpg=cpg)
        loss_gradients.update(loss_gradients_ME)

    return loss_gradients


def get_loss_gradients_weighted(model, loss_blobs, loss_weight):
    """Generate a gradient of 1 for each loss specified in 'loss_blobs'"""
    loss_gradients = {}
    for b in loss_blobs:
        loss_grad = model.net.ConstantFill(
            b, [b + '_grad'], value=1.0 * loss_weight)
        loss_gradients[str(b)] = str(loss_grad)
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #


def add_VGG16_roi_2fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    if cfg.WSL.CONTEXT:
        return add_VGG16_roi_context_2fc_head(model, blob_in, dim_in,
                                              spatial_scale)
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'pool5',
        blob_rois=prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient(l, l)

    l = model.FC(l, prefix + 'fc6', dim_in * 7 * 7, 4096)
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    l = model.FC(l, prefix + 'fc7', 4096, 4096)
    l = model.Relu(l, prefix + 'fc7')
    l = DropoutIfTraining(model, l, prefix + 'drop7', 0.5)

    return l, 4096


def add_VGG16_roi_context_2fc_head(model, blob_in, dim_in, spatial_scale):
    blobs_out = []
    # origin roi
    l = model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, 'obn_scores'], 'roi_feat_boost')

    l = model.FC(l, 'fc6', dim_in * 7 * 7, 4096)
    l = model.Relu(l, 'fc6')
    l = DropoutIfTraining(model, l, 'drop6', 0.5)
    l = model.FC(l, 'fc7', 4096, 4096)
    l = model.Relu(l, 'fc7')
    l = DropoutIfTraining(model, l, 'drop7', 0.5)

    blobs_out.append(l)

    # frame roi
    l = model.RoIFeatureTransform(
        blob_in,
        'pool5_frame',
        blob_rois='rois_frame',
        method='RoILoopPool',
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, 'obn_scores'], 'roi_feat_boost_frame')

    l = model.net.FC([l, 'fc6_w', 'fc6_b'], 'fc6_frame')
    l = model.Relu(l, 'fc6_frame')
    l = DropoutIfTraining(model, l, 'drop6_frame', 0.5)
    l = model.net.FC([l, 'fc7_w', 'fc7_b'], 'fc7_frame')
    l = model.Relu(l, 'fc7_frame')
    l = DropoutIfTraining(model, l, 'drop7_frame', 0.5)

    blobs_out.append(l)

    # context roi
    l = model.RoIFeatureTransform(
        blob_in,
        'pool5_context',
        blob_rois='rois_context',
        method='RoILoopPool',
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, 'obn_scores'], 'roi_feat_boost_context')

    l = model.net.FC([l, 'fc6_w', 'fc6_b'], 'fc6_context')
    l = model.Relu(l, 'fc6_context')
    l = DropoutIfTraining(model, l, 'drop6_context', 0.5)
    l = model.net.FC([l, 'fc7_w', 'fc7_b'], 'fc7_context')
    l = model.Relu(l, 'fc7_context')
    l = DropoutIfTraining(model, l, 'drop7_context', 0.5)

    blobs_out.append(l)

    return blobs_out, 4096


def add_ResNet_roi_0fc_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    # return 'pool5', 2048 * 7 * 7

    s = model.AveragePool('pool5', 'res5_pool', kernel=7)
    return s, 2048


def add_ResNet_roi_2fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    roi_feat_boost = model.net.RoIFeatureBoost([roi_feat, 'obn_scores'],
                                               'roi_feat_boost')

    model.FC(roi_feat_boost, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    l = DropoutIfTraining(model, 'fc6', 'drop6', 0.5)
    model.FC(l, 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    l = DropoutIfTraining(model, 'fc7', 'drop7', 0.5)
    return l, hidden_dim


def add_roi_Xconv_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    roi_feat_boost = model.net.RoIFeatureBoost([roi_feat, 'obn_scores'],
                                               'roi_feat_boost')

    current = roi_feat_boost
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'head_conv' + str(i + 1),
            dim_in,
            hidden_dim,
            3,
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    current = model.AveragePool(current, 'head_pool', kernel=roi_size)

    return current, dim_in


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    roi_feat_boost = model.net.RoIFeatureBoost([roi_feat, 'obn_scores'],
                                               'roi_feat_boost')

    current = roi_feat_boost
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'head_conv' + str(i + 1),
            dim_in,
            hidden_dim,
            3,
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current,
            'head_conv' + str(i + 1),
            dim_in,
            hidden_dim,
            3,
            group_gn=get_group_gn(hidden_dim),
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_ResNet_roi_conv5_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    l = model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient(l, l)

    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage(model, 'res5', 'pool5', 3, dim_in, 2048,
                          dim_bottleneck * 8, 1, stride_init)
    s = model.AveragePool(s, 'res5_pool', kernel=7)
    return s, 2048


def DropoutIfTraining(model, blob_in, blob_out, dropout_rate):
    """Add dropout to blob_in if the model is in training mode and
    dropout_rate is > 0."""
    if model.train and dropout_rate > 0:
        blob_out = model.Dropout(
            blob_in, blob_out, ratio=dropout_rate, is_test=False)
        return blob_out
    else:
        return blob_in
