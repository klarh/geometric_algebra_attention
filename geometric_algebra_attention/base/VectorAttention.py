import collections
import itertools
import math

from .internal import AttentionBase

class VectorAttention(AttentionBase):
    __doc__ = AttentionBase.__doc__

    # plus, minus
    _LINEAR_OPERATION_COUNT = 2

    @staticmethod
    def get_invariant_dims(rank, invariant_mode, include_normalized_products=False,
                           linear_mode='partial', linear_terms=0):
        result = 1 if rank == 1 else 2
        if invariant_mode == 'full':
            result = rank**2
        elif invariant_mode == 'partial':
            result = 2*rank - 1
        if include_normalized_products:
            result *= 2
        result += (linear_terms if rank > 1 else 0)*2
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

        linear_products = []
        if self.linear_terms and self.rank > 1:
            terms = result[self.linear_mode]
            linear_products = self.linear_terms*[0]
            kernel_weights = self.math.softmax(self.linear_kernels)

            for j, term in enumerate(terms):
                if term.rank == 1:
                    term = term._replace(products=self.algebra.vec2trivec(
                        term.products))
                if term.rank%2 != self.rank%2:
                    dual_fun = [self.algebra.bivec_dual,
                                self.algebra.trivec_dual][term.rank%2]
                    term = term._replace(products=dual_fun(term.products))

                for i in range(self.linear_terms):
                    plus = kernel_weights[i, j, 0, 1]*term.products
                    minus = kernel_weights[i, j, 1, 1]*term.products
                    linear_products[i] = linear_products[i] + plus - minus

        linear_terms = []
        invar_fun = self.algebra.custom_norm
        covar_fun = lambda x: x
        if self.rank%2:
            invar_fun = self.algebra.bivecvec_invariants
            covar_fun = self.algebra.bivecvec_covariants
        elif self.rank > 1:
            invar_fun = self.algebra.vecvec_invariants
            covar_fun = self.algebra.vecvec_covariants
        for p in linear_products:
            invariants = invar_fun(p)
            covariants = covar_fun(p)
            linear_terms.append(self.ProductType(
                self.rank, p, invariants, covariants))

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

        for key in list(result):
            result[key].extend(linear_terms)

        return result, broadcast_indices
