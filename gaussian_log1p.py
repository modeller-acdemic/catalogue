# -*- coding: utf-8 -*-
"""
Gaussian Likelihood with an identity link for a log1p-transformed response
Set a fixed variance with GaussianLogLikelihood(variance), or leave it unset to profile
the observation precision within the solver.

@author: Chris
"""

import numpy as np
import scipy

class GaussianLog1pLikelihood:

    # Initialise a Gaussian likelihood with an identity link on log1p(response)
    def __init__(self, variance=None, obs_prec_shape=1.0, obs_prec_rate=0.00005):
        self.fixed_variance = None if variance is None else float(variance)
        self.fixed_obs_tau = None if variance is None else float(1.0 / variance)
        self.obs_prec_shape = float(obs_prec_shape)
        self.obs_prec_rate = float(obs_prec_rate)
        self.last_hyper = None
        if variance is not None:
            self.last_hyper = float(self.fixed_obs_tau)
        return

    # Return the initial likelihood hyperparameter state
    def hyper_init(self, response_vals):
        if self.fixed_obs_tau is not None:
            self.last_hyper = float(self.fixed_obs_tau)
            return self.last_hyper
        response_np = self.to_numpy(response_vals)
        response_var = float(np.var(response_np, ddof=1))
        response_var = max(response_var, np.finfo(float).tiny)
        self.last_hyper = float(1.0 / response_var)
        return self.last_hyper

    # Update the likelihood hyperparameter state at the current linear predictor
    def hyper_update(self, response_vals, linear_pred, hyper=None):
        if self.fixed_obs_tau is not None:
            self.last_hyper = float(self.fixed_obs_tau)
            return self.last_hyper
        response_np = self.to_numpy(response_vals)
        mean_vals = self.mean(linear_pred)
        residual = response_np - mean_vals
        rss = float(np.dot(residual, residual))
        post_shape = self.obs_prec_shape + (0.5 * response_np.size)
        post_rate = self.obs_prec_rate + (0.5 * rss)
        obs_tau = (post_shape - 1.0) / post_rate
        obs_tau = max(obs_tau, np.finfo(float).tiny)
        self.last_hyper = float(obs_tau)
        return self.last_hyper

    # Return whether the likelihood hyperparameter state changes with the latent values
    def hyper_dynamic(self):
        return self.fixed_obs_tau is None

    # Return the log prior contribution of the likelihood hyperparameters
    def hyper_logprior(self, hyper):
        if self.fixed_obs_tau is not None:
            return 0.0
        obs_tau = self.last_hyper if hyper is None else float(hyper)
        return float((self.obs_prec_shape - 1.0) * np.log(obs_tau) - (self.obs_prec_rate * obs_tau))

    # Compute a likelihood-consistent default intercept value from the transformed response
    def default_intercept(self, response_vals):
        response_np = response_vals.to_numpy() if hasattr(response_vals, "to_numpy") else response_vals
        response_np = np.asarray(response_np)
        inter_val = np.mean(response_np)
        return float(inter_val)
    
    # Compute a likelihood-consistent lower bound for prior heuristics from the transformed response
    def prior_lower(self, response_vals):
        response_np = response_vals.to_numpy() if hasattr(response_vals, "to_numpy") else response_vals
        response_np = np.asarray(response_np)
        bound = np.percentile(response_np, 1)
        return float(bound)

    # Compute a likelihood-consistent upper bound for prior heuristics from the transformed response
    def prior_upper(self, response_vals):
        response_np = response_vals.to_numpy() if hasattr(response_vals, "to_numpy") else response_vals
        response_np = np.asarray(response_np)
        bound = np.percentile(response_np, 99)
        return float(bound)

    # Convert inputs into numpy arrays using the array interface
    def to_numpy(self, values):
        return np.asarray(values)

    # Leave the linear predictor unchanged under the identity link
    def cap_linpred(self, linear_pred):
        linpred_np = self.to_numpy(linear_pred)
        return linpred_np

    # Precompute response-only constants for reuse in likelihood evaluation
    def precompute(self, response_vals, hyper=None):
        response_np = self.to_numpy(response_vals)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        log_norm = 0.5 * np.log(obs_tau) - 0.5 * np.log(2.0 * np.pi)
        log_fact = (-0.5 * np.square(response_np) * obs_tau) + log_norm
        cache = {"log_fact": log_fact, "log_fact_sum": float(np.sum(log_fact))}
        return cache

    # Leave the linear predictor unchanged under the identity link
    def clip_linpred(self, linear_pred):
        linpred_np = self.to_numpy(linear_pred)
        return linpred_np

    # Convert the linear predictor into the mean parameter under the identity link
    def mean(self, linear_pred):
        linpred_np = self.clip_linpred(linear_pred)
        mean_vals = linpred_np
        return mean_vals

    # Compute per-observation log likelihood contributions
    def loglik(self, response_vals, linear_pred, cache=None, hyper=None):
        cache = self.precompute(response_vals, hyper) if cache is None else cache
        response_np = self.to_numpy(response_vals)
        linpred_np = self.cap_linpred(linear_pred)
        mean_vals = self.mean(linpred_np)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        log_vals = (response_np * mean_vals * obs_tau) - (0.5 * np.square(mean_vals) * obs_tau) + cache["log_fact"]
        return log_vals

    # Compute the summed log likelihood across observations
    def logsum(self, response_vals, linear_pred, cache=None, hyper=None):
        cache = self.precompute(response_vals, hyper) if cache is None else cache
        linpred_np = self.cap_linpred(linear_pred)
        mean_vals = self.mean(linpred_np)
        log_sum = self.logsum_from_mean(response_vals, linpred_np, mean_vals, cache, hyper)
        return log_sum

    # Compute the summed log likelihood when the mean has already been built
    def logsum_from_mean(self, response_vals, linear_pred, mean_vals, cache=None, hyper=None):
        cache = self.precompute(response_vals, hyper) if cache is None else cache
        response_np = self.to_numpy(response_vals)
        mean_np = self.to_numpy(mean_vals)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        log_sum = float(np.sum((response_np * mean_np * obs_tau) - (0.5 * np.square(mean_np) * obs_tau) + cache["log_fact"]))
        return log_sum

    # Compute the score with respect to the linear predictor under the identity link
    def score(self, response_vals, linear_pred, hyper=None):
        response_np = self.to_numpy(response_vals)
        mean_vals = self.mean(linear_pred)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        score_vals = (response_np - mean_vals) * obs_tau
        return score_vals

    # Compute the curvature with respect to the linear predictor under the identity link
    def curvature(self, response_vals, linear_pred, hyper=None):
        mean_vals = self.mean(linear_pred)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        curv_vals = -np.ones_like(mean_vals) * obs_tau
        return curv_vals

    # Compute the positive curvature weights used in precision construction under the identity link
    def weight(self, response_vals, linear_pred, hyper=None):
        mean_vals = self.mean(linear_pred)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        weight_vals = np.ones_like(mean_vals) * obs_tau
        return weight_vals

    # Compute the pointwise log probability mass at observed outcomes
    def logprob(self, response_vals, linear_pred, cache=None, hyper=None):
        log_vals = self.loglik(response_vals, linear_pred, cache, hyper)
        return log_vals

    # Compute the predictive CDF at observed outcomes
    def cdf(self, response_vals, linear_pred, hyper=None):
        response_np = self.to_numpy(response_vals)
        linpred_np = self.cap_linpred(linear_pred)
        mean_vals = self.mean(linpred_np)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        cdf_vals = scipy.stats.norm.cdf(response_np, mean_vals, np.sqrt(1.0 / obs_tau))
        return cdf_vals

    # Compute CDF bounds used for discrete PIT randomisation
    def pitbounds(self, response_vals, linear_pred, hyper=None):
        response_np = self.to_numpy(response_vals)
        linpred_np = self.cap_linpred(linear_pred)
        mean_vals = self.mean(linpred_np)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        sd = np.sqrt(1.0 / obs_tau)
        cdf_upper = scipy.stats.norm.cdf(response_np, mean_vals, sd)
        cdf_lower = scipy.stats.norm.cdf(response_np, mean_vals, sd)
        return cdf_lower, cdf_upper

    # Draw outcomes conditional on the linear predictor
    def draw(self, linear_pred, rng=None, hyper=None):
        rng_use = np.random if rng is None else rng
        linpred_np = self.cap_linpred(linear_pred)
        mean_vals = self.mean(linpred_np)
        draw_vals = self.draw_from_mean(mean_vals, rng=rng_use, hyper=hyper)
        return draw_vals

    # Draw outcomes when the mean has already been built
    def draw_from_mean(self, mean_vals, rng=None, hyper=None):
        rng_use = np.random if rng is None else rng
        mean_np = self.to_numpy(mean_vals)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        draw_vals = rng_use.normal(mean_np, np.sqrt(1.0 / obs_tau))
        return draw_vals

    # Provide the default intercept initialiser on the transformed-response scale
    def intercept(self, response_vals):
        response_np = self.to_numpy(response_vals)
        return float(np.mean(response_np))

    # Provide likelihood-consistent bounds for prior heuristics on the transformed-response scale
    def bounds(self, response_vals):
        response_np = self.to_numpy(response_vals)
        lower_val = np.percentile(response_np, 1)
        upper_val = np.percentile(response_np, 99)
        return float(lower_val), float(upper_val)

    # Precompute response-only constants for pointwise predictive evaluation
    def pointwise_consts(self, response_val, hyper=None):
        response_np = self.to_numpy(response_val)
        obs_tau = self.last_hyper if hyper is None else hyper
        obs_tau = self.fixed_obs_tau if self.fixed_obs_tau is not None else obs_tau
        log_norm = 0.5 * np.log(obs_tau) - 0.5 * np.log(2.0 * np.pi)
        log_fact = (-0.5 * np.square(response_np) * obs_tau) + log_norm
        cache = {"log_fact": log_fact}
        return cache

    # # Transform outcomes onto the PPC plotting scale for this likelihood
    # def ppc_transform(self, values):
    #     values_np = self.to_numpy(values)
    #     values_np = np.log1p(values_np)
    #     return values_np

    # Transform the modelled log1p response back onto the raw response scale for PPC plots
    def ppc_transform(self, values):
        values_np = self.to_numpy(values)
        values_np = np.expm1(values_np)
        return values_np
