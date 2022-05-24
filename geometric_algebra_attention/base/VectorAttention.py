import collections
import itertools
import math

from .internal import AttentionBase

class VectorAttention(AttentionBase):
    __doc__ = AttentionBase.__doc__

    @staticmethod
    def get_invariant_dims(rank, invariant_mode, include_normalized_products=False):
        result = 1 if rank == 1 else 2
        if invariant_mode == 'full':
            result = rank**2
        elif invariant_mode == 'partial':
            result = 2*rank - 1
        if include_normalized_products:
            result *= 2
        return result

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

        if self.include_normalized_products:
            for (key, series) in result.items():
                normalized_series = []
                for product in series:
                    norm = self.algebra.custom_norm(product.products)
                    normalization = self.math.clip(norm, 1e-7, math.inf)
                    scaling = 1.0/normalization
                    normalized_product = product._replace(
                        products=product.products*scaling,
                        invariants=product.invariants*scaling,
                        covariants=product.covariants*scaling)
                    normalized_series.append(normalized_product)
                series.extend(normalized_series)

        return result, broadcast_indices
