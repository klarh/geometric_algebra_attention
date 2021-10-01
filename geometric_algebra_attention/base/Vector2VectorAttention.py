
class Vector2VectorAttention:
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
