import collections
import itertools
import math

from .internal import AttentionBase

class MultivectorAttention(AttentionBase):
    __doc__ = AttentionBase.__doc__

    @staticmethod
    def get_invariant_dims(rank, invariant_mode, include_normalized_products=False):
        result = 4
        if invariant_mode == 'full':
            result = 4*rank*(rank + 1)//2
        elif invariant_mode == 'partial':
            result = 4*rank
        if include_normalized_products:
            result *= 2
        return result

    def _get_products(self, inputs):
        broadcast_indices = self._get_broadcast_indices()
        rs = [inp.positions[index] for (inp, index) in zip(inputs, broadcast_indices)]

        product_funs = lambda: itertools.chain(
            [(lambda _, x: x)], itertools.repeat(self.algebra.mvecmvec))

        result = dict(full=[], partial=[], single=[])
        for start in range(self.rank):
            series = []
            product = None
            rank = 0
            for (right, product_fn) in zip(rs[start:], product_funs()):
                rank += 1
                product = product_fn(product, right)
                invar = self.algebra.mvecmvec_invariants(product)
                covar = product
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
