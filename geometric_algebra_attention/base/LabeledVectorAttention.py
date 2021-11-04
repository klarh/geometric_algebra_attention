import math

class LabeledVectorAttention:
    """Use labels to translate one point cloud to another.

    This layer calculates a new point cloud from a set of reference
    point cloud values and coordinates, and a query set of point cloud
    values. It produces one point corresponding to each query label
    (`reduce=True`) or one point cloud, corresponding to each
    reference point, for each query label (`reduce=False`).

    :param score_net: function producing logits for the attention mechanism
    :param value_net: function producing values in the embedding dimension of the network
    :param scale_net: function producing a scalar rescaling value for the vectors produced by the network
    :param reduce: if `True`, produce a permutation-invariant result; otherwise, produce a permutation-covariant result
    :param merge_fun: Function used to merge the input values of each tuple before being passed to `join_fun`: 'mean' (no parameters) or 'concat' (learned projection for each tuple position)
    :param join_fun: Function used to join the representations of the rotation-invariant quantities (produced by `value_net`) and the tuple summary (produced by `merge_fun`): 'mean' (no parameters) or 'concat' (learned projection for each representation)
    :param rank: Degree of correlations to consider. 2 for pairwise attention, 3 for triplet-wise attention, and so on. Memory and computational complexity scales as `N**rank`
    :param invariant_mode: Type of rotation-invariant quantities to embed into the network. 'single' (use only the invariants of the final geometric product), 'partial' (use invariants for the intermediate steps to build the final geometric product), or 'full' (calculate all invariants that are possible when building the final geometric product)
    :param covariant_mode: Type of rotation-covariant quantities to use in the output calculation. 'single' (use only the vectors produced by the final geometric product), 'partial' (use all vectors for intermediate steps along the path of building the final geometric product), or 'full' (calculate the full set of vectors for the tuple)

    """
    def _build_weight_definitions(self, n_dim):
        result = super()._build_weight_definitions(n_dim)

        if self.join_fun == 'concat':
            stdev = math.sqrt(2./3./n_dim)
            result.groups['join_kernels'].append(self.WeightDefinition(
                'join_kernel_2', [n_dim, n_dim], stdev))

        return result

    def _evaluate(self, inputs, mask=None):
        (child_values, inputs) = inputs
        parsed_inputs = self._parse_inputs(inputs)
        products = self._get_product_summary(parsed_inputs)
        invar_values = self.value_net(products.summary.invariants)

        swap_i = -self.rank - 1
        swap_j = swap_i - 1
        child_expand_indices = list(products.broadcast_indices[-1])
        child_expand_indices[swap_i], child_expand_indices[swap_j] = \
            child_expand_indices[swap_j], child_expand_indices[swap_i]
        child_values = child_values[tuple(child_expand_indices)]

        joined_values = self._join_fun(child_values, invar_values, products.values)
        covariants = self._covariants(products.summary.covariants)
        new_values = covariants*self.scale_net(joined_values)

        scores = self.score_net(joined_values)
        old_shape = self.math.shape(scores)

        scores = self._mask_scores(scores, products.broadcast_indices, mask)

        attention, output = self._calculate_attention(
            scores, new_values, old_shape)

        return self.OutputType(
            attention, output, products.summary.invariants, invar_values, new_values)

    def _get_product_summary(self, inputs):
        products = super()._get_product_summary(inputs)
        broadcast_indices = []
        for idx in products.broadcast_indices:
            idx = list(idx)
            idx.insert(-1 - self.rank, None)
            broadcast_indices.append(idx)

        index = tuple([Ellipsis, None] + (self.rank + 1)*[slice(None)])
        invars = products.summary.invariants[index]
        covars = [r[index] for r in products.summary.covariants]
        new_vs = products.values[index]
        if isinstance(products.weights, int):
            weights = products.weights
        else:
            weights = products.weights[index]

        summary = products.summary._replace(invariants=invars, covariants=covars)
        return self.ProductSummaryType(summary, broadcast_indices, weights, new_vs)
