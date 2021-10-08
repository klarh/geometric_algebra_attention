import collections
import itertools
import math

HUGE_FLOAT = 1e9

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
    :param invariant_mode: Type of rotation-invariant quantities to embed into the network. 'single' (use only the invariants of the final geometric product), 'partial' (use invariants for the intermediate steps to build the final geometric product), or 'full' (calculate all invariants that are possible when building the final geometric product)

    """

    InputType = collections.namedtuple(
        'InputType', ['positions', 'values', 'weights'])

    WeightDefinition = collections.namedtuple(
        'WeightDefinition', ['name', 'shape', 'stdev'])

    WeightDefinitionSet = collections.namedtuple(
        'WeightDefinitionSet', ['groups', 'singles'])

    ProductType = collections.namedtuple(
        'ProductType', ['rank', 'products', 'invariants', 'covariants'])

    ProductSummaryType = collections.namedtuple(
        'ProductSummaryType', ['summary', 'broadcast_indices', 'weights', 'values'])

    OutputType = collections.namedtuple(
        'OutputType', ['attention', 'output', 'invariants', 'invariant_values',
                       'tuple_values'])

    def __init__(self, score_net, value_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 invariant_mode='single', covariant_mode='single'):
        self.score_net = score_net
        self.value_net = value_net
        self.reduce = reduce
        self.merge_fun = merge_fun
        self.join_fun = join_fun
        self.rank = rank
        self.invariant_mode = invariant_mode
        self.covariant_mode = covariant_mode

        for mode in [invariant_mode, covariant_mode]:
            assert mode in ['full', 'partial', 'single']

    @staticmethod
    def get_invariant_dims(rank, invariant_mode):
        if invariant_mode == 'full':
            return rank**2
        elif invariant_mode == 'partial':
            return 2*rank - 1
        return 1 if rank == 1 else 2

    @property
    def invariant_dims(self):
        return self.get_invariant_dims(self.rank, self.invariant_mode)

    def _build_weight_definitions(self, n_dim):
        result = self.WeightDefinitionSet({}, {})

        if self.merge_fun == 'concat':
            stdev = math.sqrt(2./self.rank/n_dim)
            result.groups['merge_kernels'] = [self.WeightDefinition(
                'merge_kernel_{}'.format(i), (n_dim, n_dim), stdev)
                                       for i in range(self.rank)]

        if self.join_fun == 'concat':
            # always joining neighborhood values and invariant values
            stdev = math.sqrt(2./2/n_dim)
            result.groups['join_kernels'] = [self.WeightDefinition(
                'join_kernel_{}'.format(i), (n_dim, n_dim), stdev)
                                      for i in range(2)]

        return result

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

    def _evaluate(self, inputs, mask=None):
        parsed_inputs = self._parse_inputs(inputs)
        products = self._get_product_summary(parsed_inputs)
        invar_values = self.value_net(products.summary.invariants)

        joined_values = self._join_fun(invar_values, products.values)
        new_values = products.weights*joined_values

        scores = self.score_net(joined_values)
        old_shape = self.math.shape(scores)

        scores = self._mask_scores(scores, products.broadcast_indices, mask)

        attention, output = self._calculate_attention(
            scores, new_values, old_shape)

        return self.OutputType(
            attention, output, products.summary.invariants, invar_values, new_values)

    def _get_broadcast_indices(self):
        broadcast_indices = []
        for i in range(1, self.rank + 1):
            index = [Ellipsis] + [None]*(self.rank) + [slice(None)]
            index[-i - 1] = slice(None)
            broadcast_indices.append(tuple(index))
        return broadcast_indices

    def _get_products(self, inputs):
        broadcast_indices = self._get_broadcast_indices()
        rs = [inp.positions[index] for (inp, index) in zip(inputs, broadcast_indices)]

        product_funs = lambda: itertools.chain(
            [(lambda _, x: x), self.algebra.vecvec], itertools.cycle(
                [self.algebra.bivecvec, self.algebra.trivecvec]))
        invar_funs = lambda: itertools.chain(
            [self.algebra.custom_norm, self.algebra.vecvec_invariants],
            itertools.cycle(
                [self.algebra.bivecvec_invariants, self.algebra.trivecvec_invariants]))
        covar_funs = lambda: itertools.chain(
            [(lambda x: x), self.algebra.vecvec_covariants],
            itertools.cycle(
                [self.algebra.bivecvec_covariants, self.algebra.trivecvec_covariants]))

        result = dict(full=[], partial=[], single=[])
        for start in range(self.rank):
            series = []
            product = None
            rank = 0
            for (right, product_fn, invar_fn, covar_fn) in zip(
                    rs[start:], product_funs(), invar_funs(), covar_funs()):
                rank += 1
                product = product_fn(product, right)
                invar = invar_fn(product)
                covar = covar_fn(product)
                series.append(self.ProductType(rank, product, invar, covar))
            series = list(reversed(series))
            if start == 0:
                result['single'] = [series[0]]
                result['partial'] = list(series)
            result['full'].extend(series)

        return result, broadcast_indices

    def _get_product_summary(self, inputs):
        (all_products, broadcast_indices) = self._get_products(inputs)
        covariants = [p.covariants for p in all_products[self.covariant_mode]]
        all_products = all_products[self.invariant_mode]
        weights = self._make_tuple_weights(broadcast_indices, inputs)

        if len(all_products) == 1:
            summary = all_products[0]
            summary = summary._replace(covariants=covariants)
        else:
            rank = self.rank
            products = all_products[0].products
            # this is suboptimal; basically just a workaround to abuse
            # broadcasting rules rather than having to deal with the
            # lack of simplicity when dealing with array shapes and
            # concat not automatically broadcasting for the user
            reshape_zeros = self.math.zeros_like(all_products[0].invariants[..., :1])
            # TODO replace this with a more efficient implementation
            # (split + broadcast_to + concat?)
            bcast_invars = [p.invariants + reshape_zeros for p in all_products]
            invariants = self.math.concat(bcast_invars, axis=-1)
            summary = self.ProductType(rank, products, invariants, covariants)

        vs = [inp.values[index] for (inp, index) in zip(inputs, broadcast_indices)]
        values = self._merge_fun(*vs)

        return self.ProductSummaryType(summary, broadcast_indices, weights, values)

    def _get_reduction(self):
        if self.reduce:
            dims = -(self.rank + 1)
            reduce_axes = tuple(-i - 2 for i in range(self.rank))
        else:
            dims = -self.rank
            reduce_axes = tuple(-i - 2 for i in range(self.rank - 1))
        return dims, reduce_axes

    def _join_fun(self, *args):
        if self.join_fun == 'mean':
            return sum(args)/float(len(args))
        elif self.join_fun == 'concat':
            return sum(
                [self.math.tensordot(x, b, 1) for (x, b) in
                 zip(args, self.join_kernels)])
        else:
            raise NotImplementedError()

    def _make_tuple_weights(self, broadcast_indices, inputs):
        if all(isinstance(inp.weights, int) for inp in inputs):
            return 1

        result = 1
        for (inp, broadcast) in zip(inputs, broadcast_indices):
            weights = inp.weights
            if isinstance(weights, int):
                result = result*float(weights)
                continue
            expanded_weights = weights[..., None][broadcast]
            result = result*expanded_weights
        return self.math.pow(result, 1./self.rank)

    def _mask_scores(self, scores, broadcast_indices, mask):
        if mask is not None:
            parsed_mask = self._parse_inputs(mask)
            if any(p.positions is not None for p in parsed_mask):
                masks = [p.positions[..., None][idx]
                         for (p, idx) in zip(parsed_mask, broadcast_indices)
                         if p.positions is not None]
                position_mask = sum(masks) == len(masks)
            else:
                position_mask = True
            if any(p.values is not None for p in parsed_mask):
                masks = [p.values[..., None][idx]
                         for (p, idx) in zip(parsed_mask, broadcast_indices)
                         if p.values is not None]
                value_mask = sum(masks) == len(masks)
            else:
                value_mask = True
            product_mask = self.math.logical_and(position_mask, value_mask)
            scores = self.math.where(product_mask, scores, -HUGE_FLOAT)
        return scores

    def _merge_fun(self, *args):
        if self.merge_fun == 'mean':
            return sum(args)/float(len(args))
        elif self.merge_fun == 'concat':
            return sum(
                [self.math.tensordot(x, b, 1) for (x, b) in
                 zip(args, self.merge_kernels) if x is not None])
        else:
            raise NotImplementedError()

    def _parse_inputs(self, inputs):
        result = []

        inputs = list(inputs)
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]

        for piece in (self.rank*inputs)[:self.rank]:
            if len(piece) == 2:
                (r, v) = piece
                w = 1
            elif len(piece) == 3:
                (r, v, w) = piece
            else:
                raise NotImplementedError(piece)
            result.append(self.InputType(r, v, w))
        return result
