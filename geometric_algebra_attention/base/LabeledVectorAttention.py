
class LabeledVectorAttention:
    def _build_weight_definitions(self, n_dim):
        result = super()._build_weight_definitions(n_dim)

        if self.join_fun == 'concat':
            stdev = self.math.sqrt(2./3./n_dim)
            result.groups['join_kernels'].append(self.WeightDefinition(
                'join_kernel_2', [n_dim, n_dim], stdev))

        return result

    def _evaluate(self, inputs, mask=None):
        (positions, values, child_values) = inputs
        products = self._expand_products(positions, values)
        broadcast_indices = products.broadcast_indices
        invariants = products.invariants
        neighborhood_values = self._merge_fun(*products.values)
        invar_values = self.value_net(invariants)

        swap_i = -self.rank - 1
        swap_j = swap_i - 1
        child_expand_indices = list(broadcast_indices[-1])
        child_expand_indices[swap_i], child_expand_indices[swap_j] = \
            child_expand_indices[swap_j], child_expand_indices[swap_i]
        child_values = child_values[tuple(child_expand_indices)]

        joined_values = self._join_fun(invar_values, neighborhood_values)
        covariants = self._covariants(
            products.covariants, positions, broadcast_indices,
            products.values, joined_values)
        new_values = covariants*self.scale_net(joined_values)

        scores = self.score_net(joined_values)
        old_shape = self.math.shape(scores)

        scores = self._mask_scores(scores, broadcast_indices, mask)

        attention, output = self._calculate_attention(
            scores, new_values, old_shape)

        return self.OutputType(
            attention, output, invariants, invar_values, new_values)

    def _expand_products(self, rs, vs):
        products = super()._expand_products(rs, vs)
        broadcast_indices = []
        for idx in products.broadcast_indices:
            idx = list(idx)
            idx.insert(-1 - self.rank, None)
            broadcast_indices.append(idx)

        index = tuple([Ellipsis, None] + (self.rank + 1)*[slice(None)])
        invars = products.invariants[index]
        covars = products.covariants[index]
        new_vs = [v[index] for v in products.values]

        return self.ExpandedProducts(broadcast_indices, invars, covars, new_vs)
