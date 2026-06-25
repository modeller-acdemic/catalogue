# -*- coding: utf-8 -*-
"""
Mixed Gaussian likelihood with an identity link for a log1p-transformed response.
The component weight and variances can be set directly or estimated from pilot
residuals when the class is instantiated.

@author: Chris
"""

import numpy as np
import scipy


class MixedGaussianLikelihood:

    # Fit the mixture from pilot residuals or store the supplied mixture parameters
    def __init__(self, residual_vals=None, mixture_weight=None, narrow_variance=None, wide_variance=None,
                 convergence_tolerance=1.0e-8):
        residual_supplied = residual_vals is not None

        # fit the mixture when pilot residuals were supplied
        if residual_supplied:

            # turn the residual input into a finite numpy array
            residual_np = np.asarray(residual_vals, dtype=float)
            residual_np = residual_np[np.isfinite(residual_np)]

            # use the middle 80 percent of residual magnitudes to seed the narrow component
            abs_residual = np.abs(residual_np)
            central_cut = np.quantile(abs_residual, 0.80)
            central_mask = abs_residual <= central_cut
            central_squared_residuals = np.square(residual_np[central_mask])

            # build the starting values for the two variances and the component weight
            narrow_variance = float(np.mean(central_squared_residuals)) if central_squared_residuals.size > 0 else float(np.var(residual_np, ddof=1))
            narrow_variance = max(narrow_variance, np.finfo(float).tiny)
            wide_variance = max(float(np.var(residual_np, ddof=1)), 4.0 * narrow_variance)
            mixture_weight = 0.90

            # square the residuals once so the loop can reuse them
            squared_residuals = np.square(residual_np)
            previous_log_likelihood = -np.inf

            # keep updating the mixture until the log likelihood stops moving
            while True:

                # evaluate the two normal log densities with the current parameters
                log_narrow = np.log(mixture_weight) + scipy.stats.norm.logpdf(residual_np, 0.0, np.sqrt(narrow_variance))
                log_wide = np.log(1.0 - mixture_weight) + scipy.stats.norm.logpdf(residual_np, 0.0, np.sqrt(wide_variance))
                log_mixture_density = scipy.special.logsumexp(np.stack((log_narrow, log_wide), axis=0), axis=0)

                # turn those log densities into narrow and wide component probabilities
                narrow_component_probability = np.exp(log_narrow - log_mixture_density)
                wide_component_probability = 1.0 - narrow_component_probability

                # update the mixture weight and the two variances from those probabilities
                mixture_weight = float(np.clip(np.mean(narrow_component_probability), 0.50, 0.999))
                narrow_variance = float(np.sum(narrow_component_probability * squared_residuals) / np.sum(narrow_component_probability))
                wide_variance = float(np.sum(wide_component_probability * squared_residuals) / np.sum(wide_component_probability))
                narrow_variance = max(narrow_variance, np.finfo(float).tiny)
                wide_variance = max(wide_variance, 1.0001 * narrow_variance)

                # swap the two components if the narrow one ended up wider
                if narrow_variance > wide_variance:
                    narrow_variance, wide_variance = wide_variance, narrow_variance
                    mixture_weight = 1.0 - mixture_weight

                # stop once the log likelihood change is below the requested tolerance
                log_likelihood = float(np.sum(log_mixture_density))
                if abs(log_likelihood - previous_log_likelihood) < convergence_tolerance:
                    break
                previous_log_likelihood = log_likelihood

        # otherwise, cast the supplied mixture parameters to floats
        else:
            mixture_weight = float(mixture_weight)
            narrow_variance = float(narrow_variance)
            wide_variance = float(wide_variance)

        # store the fitted or supplied mixture parameters on the class
        self.mixture_weight = float(mixture_weight)
        self.narrow_variance = float(narrow_variance)
        self.wide_variance = float(wide_variance)

        # convert the stored variances into precisions for later calculations
        self.narrow_precision = float(1.0 / self.narrow_variance)
        self.wide_precision = float(1.0 / self.wide_variance)

        # store the log component weights for the mixture-density calculations
        self.log_mixture_weight = float(np.log(self.mixture_weight))
        self.log_wide_component_weight = float(np.log(1.0 - self.mixture_weight))
        self.last_hyper = None
        return

    # Return None because this class does not initialise a free hyperparameter
    def hyper_init(self, response_vals):
        return None

    # Return None because this class does not update a free hyperparameter
    def hyper_update(self, response_vals, linear_pred, hyper=None):
        return None

    # Return False because the stored mixture parameters stay fixed during fitting
    def hyper_dynamic(self):
        return False

    # Return zero because this class does not add a hyperprior term
    def hyper_logprior(self, hyper):
        return 0.0

    # Convert the response to numpy and return its mean
    def default_intercept(self, response_vals):
        response_np = response_vals.to_numpy() if hasattr(response_vals, "to_numpy") else response_vals
        response_np = np.asarray(response_np)
        inter_val = np.mean(response_np)
        return float(inter_val)

    # Convert the response to numpy and return its 1st percentile
    def prior_lower(self, response_vals):
        response_np = response_vals.to_numpy() if hasattr(response_vals, "to_numpy") else response_vals
        response_np = np.asarray(response_np)
        bound = np.percentile(response_np, 1)
        return float(bound)

    # Convert the response to numpy and return its 99th percentile
    def prior_upper(self, response_vals):
        response_np = response_vals.to_numpy() if hasattr(response_vals, "to_numpy") else response_vals
        response_np = np.asarray(response_np)
        bound = np.percentile(response_np, 99)
        return float(bound)

    # Convert inputs into numpy arrays using the array interface
    def to_numpy(self, values):
        return np.asarray(values)

    # Convert the linear predictor to numpy without changing its values
    def cap_linpred(self, linear_pred):
        linpred_np = self.to_numpy(linear_pred)
        return linpred_np

    # Return None because this class does not build a reusable response cache
    def precompute(self, response_vals, hyper=None):
        return None

    # Convert the linear predictor to numpy without changing its values
    def clip_linpred(self, linear_pred):
        linpred_np = self.to_numpy(linear_pred)
        return linpred_np

    # Pass the clipped linear predictor through unchanged
    def mean(self, linear_pred):
        linpred_np = self.clip_linpred(linear_pred)
        mean_vals = linpred_np
        return mean_vals

    # Build the residuals, component probabilities, and log mixture density
    def mixture_terms(self, response_vals, mean_vals):

        # convert the inputs to numpy and subtract the mean from the response
        response_np = self.to_numpy(response_vals)
        mean_np = self.to_numpy(mean_vals)
        residual = response_np - mean_np

        # evaluate the narrow and wide log densities at those residuals
        log_narrow = self.log_mixture_weight + scipy.stats.norm.logpdf(residual, 0.0, np.sqrt(self.narrow_variance))
        log_wide = self.log_wide_component_weight + scipy.stats.norm.logpdf(residual, 0.0, np.sqrt(self.wide_variance))
        log_mixture_density = scipy.special.logsumexp(np.stack((log_narrow, log_wide), axis=0), axis=0)

        # convert the two log densities into narrow and wide probabilities
        narrow_component_probability = np.exp(log_narrow - log_mixture_density)
        wide_component_probability = np.exp(log_wide - log_mixture_density)
        return residual, narrow_component_probability, wide_component_probability, log_mixture_density

    # Build the mean values and return the per-row log mixture density
    def loglik(self, response_vals, linear_pred, cache=None, hyper=None):
        linpred_np = self.cap_linpred(linear_pred)
        mean_vals = self.mean(linpred_np)
        _, _, _, log_mixture_density = self.mixture_terms(response_vals, mean_vals)
        return log_mixture_density

    # Sum the per-row log likelihood values and return one number
    def logsum(self, response_vals, linear_pred, cache=None, hyper=None):
        log_sum = float(np.sum(self.loglik(response_vals, linear_pred, cache, hyper)))
        return log_sum

    # Combine the residual with the mean precision and return the score
    def score(self, response_vals, linear_pred, hyper=None):
        residual, narrow_component_probability, wide_component_probability, _ = self.mixture_terms(response_vals, self.mean(linear_pred))
        precision_mean = (narrow_component_probability * self.narrow_precision) + (wide_component_probability * self.wide_precision)
        score_vals = residual * precision_mean
        return score_vals

    # Negate the weight values and return them as curvature
    def curvature(self, response_vals, linear_pred, hyper=None):
        curv_vals = -self.weight(response_vals, linear_pred, hyper)
        return curv_vals

    # Combine the component precisions into the positive curvature weights
    def weight(self, response_vals, linear_pred, hyper=None):

        # get the residuals and the two component probabilities
        residual, narrow_component_probability, wide_component_probability, _ = self.mixture_terms(response_vals, self.mean(linear_pred))

        # average the two precisions using those component probabilities
        precision_mean = (narrow_component_probability * self.narrow_precision) + (wide_component_probability * self.wide_precision)

        # combine the first and second precision moments into the weight values
        precision_sq_mean = (narrow_component_probability * (self.narrow_precision ** 2)) + (wide_component_probability * (self.wide_precision ** 2))
        weight_vals = precision_mean - (np.square(residual) * (precision_sq_mean - np.square(precision_mean)))
        weight_vals = np.maximum(weight_vals, 1.0e-6)
        return weight_vals

    # Return the per-row log likelihood values unchanged
    def logprob(self, response_vals, linear_pred, cache=None, hyper=None):
        log_vals = self.loglik(response_vals, linear_pred, cache, hyper)
        return log_vals

    # Evaluate the two cumulative distributions and return their weighted sum
    def cdf(self, response_vals, linear_pred, hyper=None):

        # build the mean values for the current observations
        response_np = self.to_numpy(response_vals)
        linpred_np = self.cap_linpred(linear_pred)
        mean_vals = self.mean(linpred_np)

        # evaluate the narrow and wide cumulative distributions at those observations
        cdf_narrow = scipy.stats.norm.cdf(response_np, mean_vals, np.sqrt(self.narrow_variance))
        cdf_wide = scipy.stats.norm.cdf(response_np, mean_vals, np.sqrt(self.wide_variance))

        # average the two cumulative distributions using the mixture weight
        cdf_vals = (self.mixture_weight * cdf_narrow) + ((1.0 - self.mixture_weight) * cdf_wide)
        return cdf_vals

    # Build identical lower and upper probability bounds for a continuous likelihood
    def pitbounds(self, response_vals, linear_pred, hyper=None):

        # build the mean values for the current observations
        response_np = self.to_numpy(response_vals)
        linpred_np = self.cap_linpred(linear_pred)
        mean_vals = self.mean(linpred_np)

        # evaluate the narrow and wide cumulative distributions at those observations
        cdf_narrow = scipy.stats.norm.cdf(response_np, mean_vals, np.sqrt(self.narrow_variance))
        cdf_wide = scipy.stats.norm.cdf(response_np, mean_vals, np.sqrt(self.wide_variance))

        # average the two cumulative distributions using the mixture weight
        cdf_vals = (self.mixture_weight * cdf_narrow) + ((1.0 - self.mixture_weight) * cdf_wide)

        # copy the same values into the lower and upper outputs
        cdf_upper = cdf_vals
        cdf_lower = cdf_vals
        return cdf_lower, cdf_upper

    # Build the mean values and pass them to the draw helper
    def draw(self, linear_pred, rng=None, hyper=None):
        rng_use = np.random if rng is None else rng
        linpred_np = self.cap_linpred(linear_pred)
        mean_vals = self.mean(linpred_np)
        draw_vals = self.draw_from_mean(mean_vals, rng=rng_use, hyper=hyper)
        return draw_vals

    # Draw values from the narrow or wide normal component for each mean value
    def draw_from_mean(self, mean_vals, rng=None, hyper=None):

        # convert the input means to numpy and choose the random-number generator
        rng_use = np.random if rng is None else rng
        mean_np = self.to_numpy(mean_vals)

        # assign each row to the narrow or wide component
        draw_component = rng_use.uniform(size=mean_np.shape) < self.mixture_weight
        draw_vals = np.empty_like(mean_np)

        # draw the narrow-component rows and the wide-component rows separately
        draw_vals[draw_component] = rng_use.normal(mean_np[draw_component], np.sqrt(self.narrow_variance))
        draw_vals[~draw_component] = rng_use.normal(mean_np[~draw_component], np.sqrt(self.wide_variance))
        return draw_vals

    # Convert the response to numpy and return its mean
    def intercept(self, response_vals):
        response_np = self.to_numpy(response_vals)
        return float(np.mean(response_np))

    # Convert the response to numpy and return its 1st and 99th percentiles
    def bounds(self, response_vals):
        response_np = self.to_numpy(response_vals)
        lower_val = np.percentile(response_np, 1)
        upper_val = np.percentile(response_np, 99)
        return float(lower_val), float(upper_val)

    # Return None because this class does not build pointwise response constants
    def pointwise_consts(self, response_val, hyper=None):
        return None

    # Apply expm1 to move the values back onto the raw response scale
    def ppc_transform(self, values):
        values_np = self.to_numpy(values)
        values_np = np.expm1(values_np)
        return values_np
