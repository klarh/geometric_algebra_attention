from .VectorAttention import VectorAttention
from .Vector2VectorAttention import Vector2VectorAttention


class TiedVectorAttention(Vector2VectorAttention, VectorAttention):
    r"""Simultaneously calculates rotation-covariant and -invariant geometric product attention.

    This layer implements a set of geometric products over all tuples
    of length `rank`, then sums over them using an attention mechanism
    to perform a permutation-covariant (`reduce=False`) or
    permutation-invariant (`reduce=True`) result.

    Instead of returning a single rotation-invariant result (as
    :py:class:`VectorAttention`) or rotation-equivariant result (as
    :py:class:`Vector2VectorAttention`), this layer returns both
    rotation-invariant and -equivariant results simultaneously. The
    learned attention weights are "tied" in the sense that the same
    attention weights are used to reduce the set of rotation-invariant
    outputs as the rotation-equivariant outputs.

    :param score_net: function producing logits for the attention mechanism
    :param value_net: function producing values in the embedding dimension of the network
    :param scale_net: function producing a scalar rescaling value for the vectors produced by the network
    :param reduce: if `True`, produce a permutation-invariant result; otherwise, produce a permutation-covariant result
    :param merge_fun: Function used to merge the input values of each tuple before being passed to `join_fun`: 'mean' (no parameters) or 'concat' (learned projection for each tuple position)
    :param join_fun: Function used to join the representations of the rotation-invariant quantities (produced by `value_net`) and the tuple summary (produced by `merge_fun`): 'mean' (no parameters) or 'concat' (learned projection for each representation)
    :param rank: Degree of correlations to consider. 2 for pairwise attention, 3 for triplet-wise attention, and so on. Memory and computational complexity scales as `N**rank`
    :param invariant_mode: Type of rotation-invariant quantities to embed into the network. 'single' (use only the invariants of the final geometric product), 'partial' (use invariants for the intermediate steps to build the final geometric product), or 'full' (calculate all invariants that are possible when building the final geometric product)
    :param covariant_mode: Type of rotation-covariant quantities to use in the output calculation. 'single' (use only the vectors produced by the final geometric product), 'partial' (use all vectors for intermediate steps along the path of building the final geometric product), or 'full' (calculate the full set of vectors for the tuple)
    :param include_normalized_products: If True, for whatever set of products that will be computed (for a given `invariant_mode` or `covariant_mode`), also include the normalized multivector for each product
    :param convex_covariants: If True, use a convex combination of the rotation-covariant inputs (including the origin (0, 0, 0)) available, rather than an arbitrary linear combination

    """

    def _evaluate(self, inputs, mask=None):
        parsed_inputs = self._parse_inputs(inputs)
        products = self._get_product_summary(parsed_inputs)
        invar_values = self.value_net(products.summary.invariants)

        joined_values = self._join_fun(invar_values, products.values)
        covariants = self._covariants(products.summary.covariants)
        new_invar_values = products.weights * joined_values
        new_covar_values = products.weights * covariants * self.scale_net(joined_values)

        scores = self.score_net(joined_values)
        old_shape = self.math.shape(scores)

        scores = self._mask_scores(scores, products.broadcast_indices, mask)

        attention, invar_output = self._calculate_attention(
            scores, new_invar_values, old_shape
        )
        attention, covar_output = self._calculate_attention(
            scores, new_covar_values, old_shape
        )
        output = (covar_output, invar_output)

        return self.OutputType(
            attention,
            output,
            products.summary.invariants,
            invar_values,
            new_invar_values,
        )
