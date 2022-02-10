
import itertools

class Multivector2MultivectorAttention:
    """Calculate rotation-equivariant (multivector-valued) geometric product attention.

    This layer implements a set of geometric products over all tuples
    of length `rank`, then sums over them using an attention mechanism
    to perform a permutation-covariant (`reduce=False`) or
    permutation-invariant (`reduce=True`) result.

    The resulting value is a (geometric) multivector, and will rotate in
    accordance to the input vectors of the layer.

    :param score_net: function producing logits for the attention mechanism
    :param value_net: function producing values in the embedding dimension of the network
    :param scale_net: function producing a scalar rescaling value for the multivectors produced by the network
    :param reduce: if `True`, produce a permutation-invariant result; otherwise, produce a permutation-covariant result
    :param merge_fun: Function used to merge the input values of each tuple before being passed to `join_fun`: 'mean' (no parameters) or 'concat' (learned projection for each tuple position)
    :param join_fun: Function used to join the representations of the rotation-invariant quantities (produced by `value_net`) and the tuple summary (produced by `merge_fun`): 'mean' (no parameters) or 'concat' (learned projection for each representation)
    :param rank: Degree of correlations to consider. 2 for pairwise attention, 3 for triplet-wise attention, and so on. Memory and computational complexity scales as `N**rank`
    :param invariant_mode: Type of rotation-invariant quantities to embed into the network. 'single' (use only the invariants of the final geometric product), 'partial' (use invariants for the intermediate steps to build the final geometric product), or 'full' (calculate all invariants that are possible when building the final geometric product)
    :param covariant_mode: Type of rotation-covariant quantities to use in the output calculation. 'single' (use only the multivectors produced by the final geometric product), 'partial' (use all multivectors for intermediate steps along the path of building the final geometric product), or 'full' (calculate the full set of multivectors for the tuple)

    """
    def __init__(self, scale_net):
        self.scale_net = scale_net

    @property
    def input_vector_count(self):
        if self.covariant_mode == 'full':
            return self.rank*(self.rank + 1)//2
        elif self.covariant_mode == 'partial':
            return self.rank
        return 1

    def _build_weight_definitions(self, n_dim):
        result = super()._build_weight_definitions(n_dim)

        if self.input_vector_count > 1:
            result.singles['vector_kernels'] = self.WeightDefinition(
                'vector_kernels', [self.input_vector_count], .05)

        return result

    def _covariants(self, covariants_):
        if self.input_vector_count == 1:
            return covariants_[0]

        covariants = [
            vec*self.vector_kernels[i] for (i, vec) in enumerate(covariants_)]
        return sum(covariants)

    def _evaluate(self, inputs, mask=None):
        parsed_inputs = self._parse_inputs(inputs)
        products = self._get_product_summary(parsed_inputs)
        invar_values = self.value_net(products.summary.invariants)

        joined_values = self._join_fun(invar_values, products.values)
        covariants = self._covariants(products.summary.covariants)
        new_values = products.weights*covariants*self.scale_net(joined_values)

        scores = self.score_net(joined_values)
        old_shape = self.math.shape(scores)

        scores = self._mask_scores(scores, products.broadcast_indices, mask)

        attention, output = self._calculate_attention(
            scores, new_values, old_shape)

        return self.OutputType(
            attention, output, products.summary.invariants, invar_values, new_values)
