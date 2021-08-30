
class Vector2VectorAttention:
    def __init__(self, scale_net, use_product_vectors=True,
                 use_input_vectors=False, learn_vector_projection=False):
        self.scale_net = scale_net
        self.use_product_vectors = use_product_vectors
        self.use_input_vectors = use_input_vectors
        self.learn_vector_projection = learn_vector_projection

        if not hasattr(self, 'rank'):
            raise TypeError('Vector2VectorAttention __init__ called before '
                            'VectorAttention has been initialized.')

        self.input_vector_count = (
            int(self.use_product_vectors) + self.rank*self.use_input_vectors)
        if self.input_vector_count < 1:
            raise ValueError('At least one of use_product_vectors or '
                             'use_input_vectors must be True')

        self.vector_kernels = [1]
        self.vector_biases = [0]

    def _build_weight_definitions(self, n_dim):
        result = super()._build_weight_definitions(n_dim)

        if self.use_input_vectors:
            if self.learn_vector_projection:
                stdev = self.math.sqrt(2./(n_dim + 1))
                result.groups['vector_kernels'] = [self.WeightDefinition(
                    'vector_kernels_{}'.format(i), [n_dim, 1], stdev)
                    for i in range(self.input_vector_count)]
                result.singles['vector_biases'] = self.WeightDefinition(
                    'vector_biases', [self.input_vector_count], .05)
            else:
                result.singles['vector_kernels'] = self.WeightDefinition(
                    'vector_kernels', [self.input_vector_count], .05)

        return result

    def _covariants(self, covariants_, positions, broadcast_indices, expanded_values,
                    joined_values):
        covariant_vectors = []
        input_values = []
        if self.use_product_vectors:
            covariant_vectors.append(covariants_)
            input_values.append(joined_values)
        if self.use_input_vectors:
            covariant_vectors.extend(
                [positions[idx] for idx in broadcast_indices])
            input_values.extend(expanded_values)

        scalars = self.vector_kernels
        if self.learn_vector_projection:
            scalars = [self.math.tensordot(v, kernel, 1) + self.vector_biases[i]
                       for i, (v, kernel) in
                       enumerate(zip(input_values, self.vector_kernels))]

        covariants = [
            vec*scalars[i] for (i, vec) in enumerate(covariant_vectors)]
        return sum(covariants)

    def _evaluate(self, inputs, mask=None):
        parsed_inputs = self._parse_inputs(inputs)
        products = self._expand_products(
            parsed_inputs.positions, parsed_inputs.values)
        broadcast_indices = products.broadcast_indices
        invariants = products.invariants
        neighborhood_values = self._merge_fun(*products.values)
        invar_values = self.value_net(invariants)

        joined_values = self._join_fun(invar_values, neighborhood_values)
        covariants = self._covariants(
            products.covariants, parsed_inputs.positions, broadcast_indices,
            products.values, joined_values)
        tuple_weights = self._make_tuple_weights(broadcast_indices, parsed_inputs.weights)
        new_values = tuple_weights*covariants*self.scale_net(joined_values)

        scores = self.score_net(joined_values)
        old_shape = self.math.shape(scores)

        scores = self._mask_scores(scores, broadcast_indices, mask)

        attention, output = self._calculate_attention(
            scores, new_values, old_shape)

        return self.OutputType(
            attention, output, invariants, invar_values, new_values)
