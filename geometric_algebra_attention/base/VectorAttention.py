import collections
import itertools

class VectorAttention:
    """Calculates geometric product attention.

    This layer implements a set of geometric products over all tuples
    of length `rank`, then sums over them using an attention mechanism
    to perform a permutation-covariant (`reduce=False`) or
    permutation-invariant (`reduce=True`) result.

    :param score_net: function producing logits for the attention mechanism
    :param value_net: function producing values in the embedding dimension of the network
    :param reduce: if `True`, produce a permutation-invariant result; otherwise, produce a permutation-covariant result
    :param merge_fun: Function used to merge the input values of each tuple before being passed to `join_fun`: 'mean' (no parameters) or 'concat' (learned projection for each tuple position)
    :param join_fun: Function used to join the representations of the rotation-invariant quantities (produced by `value_net`) and the tuple summary (produced by `merge_fun`): 'mean' (no parameters) or 'concat' (learned projection for each representation)
    :param rank: Degree of correlations to consider. 2 for pairwise attention, 3 for triplet-wise attention, and so on. Memory and computational complexity scales as `N**rank`

    """

    InputType = collections.namedtuple(
        'InputType', ['positions', 'values', 'weights'])

    WeightDefinition = collections.namedtuple(
        'WeightDefinition', ['name', 'shape', 'stdev'])

    WeightDefinitionSet = collections.namedtuple(
        'WeightDefinitionSet', ['groups', 'singles'])

    ExpandedProducts = collections.namedtuple(
        'ExpandedProducts', ['broadcast_indices', 'invariants', 'covariants',
                             'values'])

    OutputType = collections.namedtuple(
        'OutputType', ['attention', 'output', 'invariants', 'invariant_values',
                       'tuple_values'])

    def __init__(self, score_net, value_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2):
        self.score_net = score_net
        self.value_net = value_net
        self.reduce = reduce
        self.merge_fun = merge_fun
        self.join_fun = join_fun
        self.rank = rank

    @property
    def invariant_dims(self):
        return 1 if self.rank == 1 else 2

    def _merge_fun(self, *args):
        if self.merge_fun == 'mean':
            return sum(args)/float(len(args))
        elif self.merge_fun == 'concat':
            return sum(
                [self.math.tensordot(x, b, 1) for (x, b) in zip(args, self.merge_kernels)])
        else:
            raise NotImplementedError()

    def _join_fun(self, *args):
        if self.join_fun == 'mean':
            return sum(args)/float(len(args))
        elif self.join_fun == 'concat':
            return sum(
                [self.math.tensordot(x, b, 1) for (x, b) in zip(args, self.join_kernels)])
        else:
            raise NotImplementedError()

    def _build_weight_definitions(self, n_dim):
        result = self.WeightDefinitionSet({}, {})

        if self.merge_fun == 'concat':
            stdev = self.math.sqrt(2./self.rank/n_dim)
            result.groups['merge_kernels'] = [WeightDefinition(
                'merge_kernel_{}'.format(i), (n_dim, n_dim), stdev)
                                       for i in range(self.rank)]

        if self.join_fun == 'concat':
            # always joining neighborhood values and invariant values
            stdev = self.math.sqrt(2./2/n_dim)
            result.groups['join_kernels'] = [WeightDefinition(
                'join_kernel_{}'.format(i), (n_dim, n_dim), stdev)
                                      for i in range(2)]

        return result

    def _expand_products(self, rs, vs):
        broadcast_indices = []
        for i in range(1, self.rank + 1):
            index = [Ellipsis] + [None]*(self.rank) + [slice(None)]
            index[-i - 1] = slice(None)
            broadcast_indices.append(tuple(index))

        expanded_vs = [vs[index] for index in broadcast_indices]
        expanded_rs = [rs[index] for index in broadcast_indices]

        product_funs = itertools.chain(
            [self.algebra.vecvec], itertools.cycle(
                [self.algebra.bivecvec, self.algebra.trivecvec]))
        invar_funs = itertools.chain(
            [self.algebra.vecvec_invariants],
            itertools.cycle(
                [self.algebra.bivecvec_invariants, self.algebra.trivecvec_invariants]))
        covar_funs = itertools.chain(
            [self.algebra.vecvec_covariants],
            itertools.cycle(
                [self.algebra.bivecvec_covariants, self.algebra.trivecvec_covariants]))

        left = expanded_rs[0]

        invar_fn = self.algebra.custom_norm
        covar_fn = lambda x: x
        for (product_fn, invar_fn, covar_fn, right) in zip(
                product_funs, invar_funs, covar_funs, expanded_rs[1:]):
            left = product_fn(left, right)

        invar = invar_fn(left)
        covar = covar_fn(left)

        return self.ExpandedProducts(broadcast_indices, invar, covar, expanded_vs)

    def _mask_scores(self, scores, broadcast_indices, mask):
        if mask is not None:
            parsed_mask = self._parse_inputs(mask)
            position_mask = parsed_mask.positions
            value_mask = parsed_mask.values
            if position_mask is not None:
                position_mask = position_mask[..., None]
                position_mask = self.math.all([position_mask[idx] for idx in broadcast_indices[:-1]], axis=0)
            else:
                position_mask = True
            if value_mask is not None:
                value_mask = value_mask[..., None]
                value_mask = self.math.all([value_mask[idx] for idx in broadcast_indices[:-1]], axis=0)
            else:
                value_mask = True
            product_mask = self.math.logical_and(position_mask, value_mask)
            scores = self.math.where(product_mask, scores, -HUGE_FLOAT)
        return scores

    def _evaluate(self, inputs, mask=None):
        parsed_inputs = self._parse_inputs(inputs)
        products = self._expand_products(
            parsed_inputs.positions, parsed_inputs.values)
        broadcast_indices = products.broadcast_indices
        invariants = products.invariants
        neighborhood_values = self._merge_fun(*products.values)
        invar_values = self.value_net(invariants)

        joined_values = self._join_fun(invar_values, neighborhood_values)
        tuple_weights = self._make_tuple_weights(broadcast_indices, parsed_inputs.weights)
        new_values = tuple_weights*joined_values

        scores = self.score_net(joined_values)
        old_shape = self.math.shape(scores)

        scores = self._mask_scores(scores, broadcast_indices, mask)

        attention, output = self._calculate_attention(
            scores, new_values, old_shape)

        return self.OutputType(
            attention, output, invariants, invar_values, new_values)

    def _get_reduction(self):
        if self.reduce:
            dims = -(self.rank + 1)
            reduce_axes = tuple(-i - 2 for i in range(self.rank))
        else:
            dims = -self.rank
            reduce_axes = tuple(-i - 2 for i in range(self.rank - 1))
        return dims, reduce_axes

    def _calculate_attention(self, scores, values, old_shape):
        dims, reduce_axes = self._get_reduction()

        shape = self.math.concat(
            [self.math.asarray(old_shape[:dims]),
             self.math.product(self.math.asarray(old_shape[dims:]), 0,
                               keepdims=True)], -1)
        scores = self.math.reshape(scores, shape)
        attention = self.math.reshape(self.math.softmax(scores), old_shape)
        output = self.math.sum(attention*values, reduce_axes)

        return attention, output

    def _make_tuple_weights(self, broadcast_indices, weights):
        if isinstance(weights, int):
            return weights
        expanded_weights = [weights[..., None][idx] for idx in broadcast_indices]
        result = 1
        for w in expanded_weights:
            result = result*w
        return self.math.pow(result, 1./self.rank)

    def _parse_inputs(self, inputs):
        if len(inputs) == 2:
            (r, v) = inputs
            w = 1
        elif len(inputs) == 3:
            (r, v, w) = inputs
        else:
            raise NotImplementedError(inputs)
        return self.InputType(r, v, w)
