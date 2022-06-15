import collections
import itertools
import math

HUGE_FLOAT = 1e9

class Namespace:
    def __init__(self, **kwargs):
        for (name, val) in kwargs.items():
            setattr(self, name, val)

class AttentionBase:
    r"""Calculates geometric product attention.

    This layer implements a set of geometric products over all tuples
    of length `rank`, then sums over them using an attention mechanism
    to perform a permutation-covariant (`reduce=False`) or
    permutation-invariant (`reduce=True`) result.

    :py:class:`VectorAttention` calculates attention using geometric
    products of input vectors, whereas
    :py:class:`MultivectorAttention` calculates attention over
    geometric products of input multivectors. Other arguments remain
    the same between the two classes.

    The overall calculation proceeds as follows. First, geometric
    products :math:`p_{ijk...}` are calculated for all tuples of the
    given `rank` given input (multi-)vectors
    :math:`\vec{r}_i`. Rotation-invariant attributes of the geometric
    products, :math:`q_{ijk...}`, are calculated from each
    product. Summary representations of the tuple :math:`v_{ijk...}`
    are computed using the given joining- and merging-functions
    :math:`\mathcal{J}` and :math:`\mathcal{M}`, per-bond embeddings
    :math:`v_i`, and as a value-generating function
    :math:`\mathcal{V}`. Attention weights are generated by softmax
    over score logits generated by a score-generating function
    :math:`\mathcal{S}`, before the output value is generated using a
    simple sum as follows:

    .. math::

        p_{ijk...} &= \vec{r}_i\vec{r}_j\vec{r}_k ... \\
        q_{ijk...} &= \text{invariants}(p_{ijk...}) \\
        v_{ijk...} &= \mathcal{J}(\mathcal{V}(q_{ijk...}), \mathcal{M}(v_i, v_j, v_k, ...)) \\
        w_{ijk...} &= \operatorname*{\text{softmax}}\limits_{jk...}(\mathcal{S}(v_{ijk...})) \\
        y_i &= \sum\limits_{jk...} w_{ijk...} v_{ijk...}

    **Permutation equivariance.** The attention weight softmax and sum
    can be either performed over the remaining indices :math:`jk...`
    or all indices :math:`ijk...` according to whether a
    permutation-equivariant (`reduce=False`) or
    permutation-invariant (`reduce=True`) result is
    desired. Permutation-equivariant layers consume a point cloud
    and produce a point cloud's worth of values, while
    permutation-invariant layers consume a point cloud and produce a
    summary value over the entire point cloud.

    **Geometric product modes.** Different sets of geometric products
    can be considered for the calculation of intermediates within the
    layer. When `invariant_mode='single'`, only the final geometric
    product :math:`p_{ijk...}` is used to calculate rotation-invariant
    features for the geometric representation of the tuple. For
    `invariant_mode='partial'`, all intermediate calculations on the
    way to the final product are used and concatenated:
    :math:`\vec{r}_i, \vec{r}_i\vec{r}_j, \vec{r}_i\vec{r}_j\vec{r}_k,
    ...`. For `invariant_mode='full'`, all combinations of geometric
    products for the tuple up to the given rank are used and
    concatenated: :math:`\vec{r}_i, \vec{r}_j, \vec{r}_i\vec{r}_j,
    ...`. While non-`single` modes can introduce some redundancy into
    the calculation, they may also simplify the functions the layer
    must learn.

    :param score_net: function producing logits for the attention mechanism
    :param value_net: function producing values in the embedding dimension of the network
    :param reduce: if `True`, produce a permutation-invariant result; otherwise, produce a permutation-covariant result
    :param merge_fun: Function used to merge the input values of each tuple before being passed to `join_fun`: 'mean' (no parameters) or 'concat' (learned projection for each tuple position)
    :param join_fun: Function used to join the representations of the rotation-invariant quantities (produced by `value_net`) and the tuple summary (produced by `merge_fun`): 'mean' (no parameters) or 'concat' (learned projection for each representation)
    :param rank: Degree of correlations to consider. 2 for pairwise attention, 3 for triplet-wise attention, and so on. Memory and computational complexity scales as `N**rank`
    :param invariant_mode: Type of rotation-invariant quantities to embed into the network. 'single' (use only the invariants of the final geometric product), 'partial' (use invariants for the intermediate steps to build the final geometric product), or 'full' (calculate all invariants that are possible when building the final geometric product)
    :param include_normalized_products: If True, for whatever set of products that will be computed (for a given `invariant_mode`), also include the normalized multivector for each product

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
                 invariant_mode='single', covariant_mode='single',
                 include_normalized_products=False):
        self.score_net = score_net
        self.value_net = value_net
        self.reduce = reduce
        self.merge_fun = merge_fun
        self.join_fun = join_fun
        self.rank = rank
        self.invariant_mode = invariant_mode
        self.covariant_mode = covariant_mode
        self.include_normalized_products = include_normalized_products

        for mode in [invariant_mode, covariant_mode]:
            assert mode in ['full', 'partial', 'single']

    @property
    def invariant_dims(self):
        return self.get_invariant_dims(
            self.rank, self.invariant_mode, self.include_normalized_products)

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
        for i in range(self.rank):
            index = [Ellipsis] + [None]*(self.rank) + [slice(None)]
            index[i - self.rank - 1] = slice(None)
            broadcast_indices.append(tuple(index))
        return broadcast_indices

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
                masks = [self.math.bool_to_int(p.positions[..., None][idx])
                         for (p, idx) in zip(parsed_mask, broadcast_indices)
                         if p.positions is not None]
                position_mask = sum(masks) == len(masks)
            else:
                position_mask = True
            if any(p.values is not None for p in parsed_mask):
                masks = [self.math.bool_to_int(p.values[..., None][idx])
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

class LabeledAttentionBase:
    r"""Use labels to translate one point cloud to another.

    This layer calculates a new point cloud from a set of reference
    point cloud values and coordinates, and a query set of point cloud
    values. It produces one point corresponding to each query label
    (`reduce=True`) or one point cloud, corresponding to each
    reference point, for each query label (`reduce=False`).

    This layer augments the per-tuple representation with an
    additional single set of labeled point values :math:`c_l`. This
    type of calculation can be used to implement translation between
    two sets of point clouds, for example. The overall attention
    scheme is as follows; consult :py:class:`VectorAttention` for
    elaboration on arguments.

    .. math::

        p_{ijk...} &= \vec{r}_i\vec{r}_j\vec{r}_k ... \\
        q_{ijk...} &= \text{invariants}(p_{ijk...}) \\
        v_{l, ijk...} &= \mathcal{J}(c_l, \mathcal{V}(q_{ijk...}), \mathcal{M}(v_i, v_j, v_k, ...)) \\
        w_{l, ijk...} &= \operatorname*{\text{softmax}}\limits_{ijk...}(\mathcal{S}(v_{l, ijk...})) \\
        y_l &= \sum\limits_{ijk...} w_{l, ijk...} v_{l, ijk...}

    :param score_net: function producing logits for the attention mechanism
    :param value_net: function producing values in the embedding dimension of the network
    :param scale_net: function producing a scalar rescaling value for the vectors produced by the network
    :param reduce: if `True`, produce a permutation-invariant result; otherwise, produce a permutation-covariant result
    :param merge_fun: Function used to merge the input values of each tuple before being passed to `join_fun`: 'mean' (no parameters) or 'concat' (learned projection for each tuple position)
    :param join_fun: Function used to join the representations of the rotation-invariant quantities (produced by `value_net`) and the tuple summary (produced by `merge_fun`): 'mean' (no parameters) or 'concat' (learned projection for each representation)
    :param rank: Degree of correlations to consider. 2 for pairwise attention, 3 for triplet-wise attention, and so on. Memory and computational complexity scales as `N**rank`
    :param invariant_mode: Type of rotation-invariant quantities to embed into the network. 'single' (use only the invariants of the final geometric product), 'partial' (use invariants for the intermediate steps to build the final geometric product), or 'full' (calculate all invariants that are possible when building the final geometric product)
    :param covariant_mode: Type of rotation-covariant quantities to use in the output calculation. 'single' (use only the vectors produced by the final geometric product), 'partial' (use all vectors for intermediate steps along the path of building the final geometric product), or 'full' (calculate the full set of vectors for the tuple)
    :param include_normalized_products: If True, for whatever set of products that will be computed (for a given `invariant_mode`), also include the normalized multivector for each product

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

        swap_i = -self.rank - 2
        swap_j = -2
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

    def _mask_scores(self, scores, broadcast_indices, mask):
        masked_scores = scores
        if mask is not None:
            (child_mask, other_mask) = mask
            masked_scores = AttentionBase._mask_scores(
                self, scores, broadcast_indices, other_mask)
            if child_mask is not None:
                index = tuple([Ellipsis, slice(None)] + (self.rank + 1)*[None])
                masked_scores = self.math.where(child_mask[index], masked_scores, -HUGE_FLOAT)
        return masked_scores
