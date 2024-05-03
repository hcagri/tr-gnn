import torch
from torch import nn


class HICLNet(nn.Module):
    """
    Hierarchical network that contains all layers
    """
    def __init__(self, submodel_type, submodel_params, hicl_depth, use_motion, use_reid_edge, use_pos_edge,
                 share_weights, edge_level_embed, node_level_embed):
        """
        :param model_type: Network to use at each layer
        :param model_params: Parameters of the model for each layer
        :param depth: Number of layers in the hierarchical model
        """
        super(HICLNet, self).__init__()
        
        for per_layer_params in (use_motion, use_reid_edge, use_pos_edge):
            assert hicl_depth == len(per_layer_params), f"{hicl_depth }, {per_layer_params}"

        assert share_weights in ('none', 'all_but_first', 'all') 
        _SHARE_WEIGHTS_IDXS = {'none': range(hicl_depth), 
                               'all_but_first':[0]+ (hicl_depth - 1)*[1], # e.g. [0, 1, 1, 1]
                               'all': hicl_depth*[0]} # e.g. [0, 0, 0, 0]

        layer_idxs = _SHARE_WEIGHTS_IDXS[share_weights]

        layers =  [submodel_type(submodel_params)
                   for i in range(hicl_depth)]
        self.layers = nn.ModuleList([layers[idx] for idx in layer_idxs])

        self.token = nn.Embedding(hicl_depth, submodel_params['transformer_params']['d_model'])
                    

    def forward(self, data, ix_layer):
        """
        Forward pass with the self.layers[ix_layer]
        """

        depth_token = self.token.weight[ix_layer]

        return self.layers[ix_layer](data, depth_token, ix_layer)
