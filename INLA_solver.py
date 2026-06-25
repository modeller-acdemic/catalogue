# -*- coding: utf-8 -*-
"""
INLA solver in OOP format
Based on Rue et al. (2009) and Fong et al. (2010)

This version is ideal for Poisson likelihoods on the log-link scale

"""

import pandas as pd
import numpy as np
import math
import os
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.ticker
import scipy
from tqdm import tqdm

np.seterr(over='ignore', invalid='ignore')  # silence annoying invalid val warnings that pop up sometimes

"""
Global hyperparameters - feel free to leave these at default - they currently produce results that match the R-INLA solver

"""

# set a random seed
np.random.seed(42)

# precision hyperparameters
gamma_shape = 1  # gamma prior shape for the random-effect precision - used R-INLA default
gamma_rate = 0.00005  # gamma prior rate for the random-effect precision - used R-INLA default
tau_start = math.exp(4)  # starting value for the random-effect precision during mode search - used R-INLA default
mode_step = 0.005  # step size used when searching for the posterior mode of the random-effect precision - used R-INLA default
max_iters = 1000  # maximum number of mode search steps for the random-effect precision - used R-INLA default
latent_max_iters = 30  # maximum number of Newton steps when optimising the latent field for a fixed precision - set it to a reasonable value
latent_tol = 0.005  # convergence tolerance for updating intercept, coefficients and random intercepts at a fixed precision - used R-INLA default
probe_step = 0.005  # step size used to estimate how sharply peaked the precision posterior is around its mode - set it to a reasonable value

# search hyperparameters
grid_dz = 0.75  # grid spacing for candidate precision values around the posterior mode (higher is coarser and faster, lower is finer and slower) - used R-INLA default
grid_max = 4  # maximum number of grid steps searched on each side of the posterior mode (higher explores further into the tails but is slower) - used R-INLA default
stop = 6  # stopping criterion for truncating the search grid by log density drop from the mode (higher includes more tail mass but is slower) - used R-INLA default
gh_count = 5  # Gauss-Hermite evaluation points per marginal correction (higher is more accurate but is slower) - used R-INLA default

# credible interval mass
cred_mass = 0.95  # 95% CI

# batch size used when summarising fitted means over all observations
summary_batch_size = 8192

# plotting hyperparameters
density_sample_size = 2000  # number of values used to estimate each posterior predictive overlay density
simulated_sets = 500  # number of simulated uniform datasets shown in the LOO-PIT plot

class inlaSolver:

    # Construct the solver with model objects and the dataset
    def __init__(self, linear_model, likelihood_model, prior, intercept_precision_prior, fixed_precision_prior,
                 fixed_vals, response_vals, rand_vals=None):

        # store the main model objects
        self.linear_model = linear_model
        self.likelihood_model = likelihood_model
        self.prior = prior
        self.intercept_precision_prior = intercept_precision_prior
        self.fixed_precision_prior = fixed_precision_prior

        # Store the input data in its original containers
        self.fixed_effects = fixed_vals
        self.response = response_vals

        # Store fixed effects and response as pandas objects so downstream methods can call to_numpy()
        self.fixed_vals = fixed_vals
        self.response_vals = response_vals
        self.response_param_start = self.likelihood_model.hyper_init(response_vals)

        # Build the fixed-effects design matrix with an intercept column
        fixed_np = fixed_vals.to_numpy()
        self.fixed_design = np.concatenate((np.ones((fixed_np.shape[0], 1)), fixed_np), axis=1)

        # Store common sizes used across solver routines
        self.obs_count = int(self.fixed_design.shape[0])
        self.fixed_count = int(self.fixed_design.shape[1])

        # Prepare and store random-effects indexing when random effects are present
        if rand_vals is None:
            self.rand_vals = None
            self.rand_count = 0
        else:
            group_index, group_count = self.linear_model.prepare_groups(rand_vals)
            self.rand_vals = pd.Series(group_index.astype(int), index=fixed_vals.index)
            self.rand_count = int(group_count)

        # Build a deterministic cache location for this model/data combination
        self.model_cache_dir = os.path.join(os.path.dirname(__file__), "model-cache")
        self.model_cache_path = os.path.join(self.model_cache_dir, "inla-cache.pkl")

        return

    # store the fitted INLA state on the solver instance
    def store_inla_state(self, posterior_intermean, posterior_intervar, posterior_fixedmeans, posterior_fixedvars,
                         posterior_fixedcov, posterior_randmeans, posterior_randvars, tau_candidates, tau_weights,
                         fixed_mix, rand_mix, response_params):

        self.posterior_intermean = posterior_intermean
        self.posterior_intervar = posterior_intervar
        self.posterior_fixedmeans = posterior_fixedmeans
        self.posterior_fixedvars = posterior_fixedvars
        self.posterior_fixedcov = posterior_fixedcov
        self.posterior_randmeans = posterior_randmeans
        self.posterior_randvars = posterior_randvars
        self.tau_candidates = tau_candidates
        self.tau_weights = tau_weights
        self.fixed_mix = fixed_mix
        self.rand_mix = rand_mix
        self.response_params = response_params

        return

    # write the fitted INLA state to a cache file so later crashes do not lose the expensive solve
    def save_inla_cache(self):

        os.makedirs(self.model_cache_dir, exist_ok=True)

        # collect the fitted INLA state that should survive a later crash in the post-fit summary code
        cache_payload = {
            "posterior_intermean": self.posterior_intermean,
            "posterior_intervar": self.posterior_intervar,
            "posterior_fixedmeans": self.posterior_fixedmeans,
            "posterior_fixedvars": self.posterior_fixedvars,
            "posterior_fixedcov": self.posterior_fixedcov,
            "posterior_randmeans": self.posterior_randmeans,
            "posterior_randvars": self.posterior_randvars,
            "tau_candidates": self.tau_candidates,
            "tau_weights": self.tau_weights,
            "fixed_mix": self.fixed_mix,
            "rand_mix": self.rand_mix,
            "response_params": self.response_params,
        }

        # write the cache payload to a temporary file first so an interrupted save does not corrupt the cache
        temp_cache_path = f"{self.model_cache_path}.{os.getpid()}.tmp"
        with open(temp_cache_path, "wb") as cache_file:
            pickle.dump(cache_payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)

        # replace the cache file once the temporary file has been written successfully
        os.replace(temp_cache_path, self.model_cache_path)

        print(f"Saved INLA cache: {self.model_cache_path}")

        return

    # load a matching fitted INLA state from the cache if one exists
    def load_inla_cache(self):

        # stop immediately when the cache folder does not exist yet
        if not os.path.isdir(self.model_cache_dir):
            return False

        # try the fixed cache name first, then fall back to any legacy cache files in the folder
        cache_files = []
        if os.path.isfile(self.model_cache_path):
            cache_files.append(self.model_cache_path)

        # keep legacy cache files available so old long-running solves remain loadable after code edits
        for cache_name in sorted(os.listdir(self.model_cache_dir)):
            cache_path = os.path.join(self.model_cache_dir, cache_name)
            if (cache_path != self.model_cache_path) and cache_name.endswith(".pkl") and os.path.isfile(cache_path):
                cache_files.append(cache_path)

        # stop immediately when there are no cache files available to try
        if len(cache_files) == 0:
            return False

        # load the most recent cache file in the folder
        cache_path = max(cache_files, key=os.path.getmtime)
        with open(cache_path, "rb") as cache_file:
            cache_payload = pickle.load(cache_file)

        # keep compatibility with older cache files saved before the response-parameter rename
        response_params = cache_payload["response_params"] if "response_params" in cache_payload else cache_payload["lik_hyper_mix"]

        # copy the loaded INLA state onto the solver
        self.store_inla_state(cache_payload["posterior_intermean"], cache_payload["posterior_intervar"],
                              cache_payload["posterior_fixedmeans"], cache_payload["posterior_fixedvars"],
                              cache_payload["posterior_fixedcov"], cache_payload["posterior_randmeans"],
                              cache_payload["posterior_randvars"], cache_payload["tau_candidates"],
                              cache_payload["tau_weights"], cache_payload["fixed_mix"], cache_payload["rand_mix"],
                              response_params)

        print(f"Loaded INLA cache: {cache_path}")
        return True

    # helper to set the prior lower bound with a default at 5% the response
    def lower_bound(self, values):
        # return the likelihood-specific lower bound on the correct link scale
        bound = self.likelihood_model.prior_lower(values)
        return float(bound)

    # helper to set the prior upper bound with a default at 95% the response
    def upper_bound(self, values):
        # return the likelihood-specific upper bound on the correct link scale
        bound = self.likelihood_model.prior_upper(values)
        return float(bound)

    # the default intercept term
    def intercept(self, response_vals):
        mean = self.likelihood_model.default_intercept(response_vals)
        return float(mean)

    # unnormalised log posterior density to evaluate tau
    def tau_upd(self, fixed_vals, response_vals, rand_vals, inter, coeffs, rand_inters, tau, log_fact, response_param=None):

        # add the group offset to the fixed effects predictor
        linpred = inter + (fixed_vals @ coeffs) + self.linear_model.random_offset(rand_inters, rand_vals)

        # update the response parameter at the current linear predictor
        response_param = self.likelihood_model.hyper_update(response_vals, linpred, response_param)

        # rebuild response-only constants when the response parameter changes with the latent values
        if self.likelihood_model.hyper_dynamic():
            log_fact = self.likelihood_model.precompute(response_vals, response_param)

        # compute the log likelihood using the likelihood model and its cached constants
        likelihood = self.likelihood_model.logsum(response_vals, linpred, log_fact, response_param)

        # posterior factor for the random effects
        log_rand = self.linear_model.rand_logprior(rand_inters, tau)

        # build fixed-effect prior contributions
        prior_inter = self.prior.log_prior(inter, response_vals, precision=self.intercept_precision_prior)

        # build coefficient prior contributions
        prior_coeffs = self.prior.log_prior(coeffs, response_vals, fixed_vals=fixed_vals,
                                            precision=self.fixed_precision_prior)

        # log joint density for response and latent terms given tau
        density = likelihood + prior_inter + prior_coeffs + log_rand + self.likelihood_model.hyper_logprior(response_param)
        return density

    # reduced Newton system at the current latent values
    def newton_system(self, fixed_vals, response_vals, rand_vals, inter, coeffs, rand_inters, tau, response_param=None):

        # linear predictor for each observation
        linpred = inter + (fixed_vals @ coeffs) + rand_inters[rand_vals]

        # update the response parameter at the current linear predictor
        response_param = self.likelihood_model.hyper_update(response_vals, linpred, response_param)

        # positive curvature weights used by the Newton system
        weight_vect = self.likelihood_model.weight(response_vals, linpred, response_param)

        # scoring system used by the Newton system
        score_vect = self.likelihood_model.score(response_vals, linpred, response_param)

        # gradient element for fixed intercepts
        gradient = np.sum(score_vect)
        gradient_coeffs = fixed_vals.T @ score_vect
        gradient_fixed = np.concatenate(([gradient], gradient_coeffs))

        # Hessian blocks for fixed intercept and coefficients
        hess_inter = -np.sum(weight_vect)
        hess_inter_coeffs = -(fixed_vals.T @ weight_vect)

        # reuse the weight-scaled design matrix for both the fixed Hessian and the grouped cross terms
        weighted_fixed = np.asarray(weight_vect).reshape(-1, 1) * fixed_vals
        hess_coeffs = -(fixed_vals.T @ weighted_fixed)

        # block matrix assembly for full fixed Hessian
        hess_fixed = np.empty((1 + hess_coeffs.shape[0], 1 + hess_coeffs.shape[1]))
        hess_fixed[0, 0] = hess_inter
        hess_fixed[0, 1:] = hess_inter_coeffs
        hess_fixed[1:, 0] = hess_inter_coeffs
        hess_fixed[1:, 1:] = hess_coeffs

        # add the normal prior gradient for fixed intercept
        gradient_fixed[0] = gradient_fixed[0] - (self.intercept_precision_prior * inter)

        # add the normal prior gradient for fixed-effect coefficients
        gradient_fixed[1:] = gradient_fixed[1:] - (self.fixed_precision_prior * coeffs)

        # add the normal prior Hessian for fixed intercept
        hess_fixed[0, 0] = hess_fixed[0, 0] - self.intercept_precision_prior

        # add the normal prior Hessian for fixed-effect coefficients
        coeff_count = hess_fixed.shape[0] - 1
        coeff_block = hess_fixed[1:, 1:]
        coeff_block[np.diag_indices(coeff_count)] = coeff_block[
                                                        np.diag_indices(coeff_count)] - self.fixed_precision_prior

        # build the fixed-effect precision block from the log posterior curvature
        fixed_precision = -hess_fixed

        # calculate per-group sums needed by the random-intercept block
        weights_sum, score_sum = self.linear_model.group_summaries(weight_vect, score_vect, rand_vals, len(rand_inters))

        # build the random-intercept precision diagonal from the log posterior curvature
        random_precision = self.linear_model.random_precision_diag(weights_sum, tau)
        random_precision_inv = 1 / random_precision

        # build the fixed-random cross precision block
        cross_precision = self.linear_model.cross_precision_terms(self.fixed_design, weight_vect, rand_vals,
                                                                  len(rand_inters))

        # build the random-intercept gradient from the log posterior score
        gradient_rand = self.linear_model.random_score(score_sum, rand_inters, tau)

        # reduced system reused by both paths
        cross_scaled = cross_precision * random_precision_inv
        reduced_precision = fixed_precision - (cross_scaled @ cross_precision.T)
        gradient_reduced = gradient_fixed - (cross_scaled @ gradient_rand)

        return linpred, weight_vect, score_vect, response_param, fixed_precision, cross_precision, random_precision_inv, gradient_rand, reduced_precision, gradient_reduced

    # models covariance outputs at the latent mode
    def latent_cov(self, fixed_vals, response_vals, rand_vals, inter, coeffs, rand_inters, tau, response_param):

        # rebuild linear predictor using final parameters
        linpred = inter + (fixed_vals @ coeffs) + rand_inters[rand_vals]

        # build the positive curvature weights used by the covariance stage
        mean_vect = self.likelihood_model.weight(response_vals, linpred, response_param)
        hess_inter = -np.sum(mean_vect)

        # cross curvature and coefficients
        hess_inter_coeffs = -(fixed_vals.T @ mean_vect)
        hess_coeffs = -(fixed_vals.T @ (np.asarray(mean_vect).reshape(-1, 1) * fixed_vals))
        inter_block = np.array([[hess_inter]])

        # reshape cross curvature terms to fill block matrix
        inter_row = hess_inter_coeffs.reshape(1, -1)
        inter_col = hess_inter_coeffs.reshape(-1, 1)
        hess_fixed = np.concatenate(
            (np.concatenate((inter_block, inter_row), axis=1), np.concatenate((inter_col, hess_coeffs), axis=1)),
            axis=0)

        # build the fixed-effect precision block from the likelihood curvature
        fixed_precision = -hess_fixed

        # add the normal prior precision to the fixed-effect precision
        prior_diag = np.concatenate(
            ([self.intercept_precision_prior], self.fixed_precision_prior * np.ones(fixed_precision.shape[0] - 1)))
        fixed_precision = fixed_precision + np.diag(prior_diag)

        # reduce the fixed-effect precision using the Schur complement identity
        random_precision, reduced_precision, cross_precision = self.reduce_precision(fixed_precision, fixed_vals,
                                                                                     rand_vals, mean_vect, tau,
                                                                                     len(rand_inters))

        # invert marginal fixed precision to get the fixed-effect covariance matrix
        cov_fixed = scipy.linalg.inv(reduced_precision, check_finite=False)

        # make it symmetrical to remove asymmetry from inversion
        cov_fixed = 0.5 * (cov_fixed + cov_fixed.T)

        # extract variance and coefficients
        var_inter = float(cov_fixed[0, 0])
        var_coeffs = np.diag(cov_fixed)[1:]

        # random-effect variances from the marginal covariance diagonal
        random_precision_inv = 1 / random_precision
        cross_scaled = cross_precision * random_precision_inv
        rand_quad = cov_fixed @ cross_scaled
        var_rand = random_precision_inv + np.sum(cross_scaled * rand_quad, axis=0)

        return var_inter, var_coeffs, var_rand, cov_fixed

    # inner optimiser for INLA that updates the latent parameters
    def optimise_latent(self, fixed_vals, response_vals, rand_vals, tau, tol, max_iters, inter_start=None,
                        coeffs_start=None, rand_inters_start=None, clamp_index=None, clamp_val=None, mode_only=False):

        # check for perfect multicollinearity in fixed effects
        if mode_only == False:
            design_rank = np.linalg.matrix_rank(fixed_vals)
            cols_count = fixed_vals.shape[1]
            if design_rank < cols_count:
                raise np.linalg.LinAlgError(
                    "Perfect multicollinearity in fixed effects. At least one column is an exact linear combination of others.")

        # initialise the response parameter at default
        inter = 0.0
        if inter_start is None:
            inter = self.intercept(response_vals)
        else:
            inter = inter_start

        # initialise default fixed-effect coefficients at zero
        coeffs = np.zeros(fixed_vals.shape[1], dtype=float) if coeffs_start is None else np.asarray(coeffs_start,
                                                                                                    dtype=float)  # don't remove this data type declaration - it's needed downstream

        # initialise default random intercepts at zero
        rand_inters = np.zeros(int(np.max(rand_vals)) + 1, dtype=float) if rand_inters_start is None else np.asarray(
            rand_inters_start, dtype=float)
        response_param = self.response_param_start

        # run Newton update with one coordinate held fixed
        if clamp_index is not None:

            # boolean array for coordinates to change when solving linear system
            free_mask = np.ones(1 + len(coeffs) + len(rand_inters), dtype=bool)

            # excluded fixed coordinates when solving linear system
            free_mask[clamp_index] = False

            # set fixed value before optimisation
            if clamp_index == 0:  # intercept
                inter = clamp_val
            elif clamp_index < 1 + len(coeffs):  # coefficient

                # apply intercept offset
                coeffs[clamp_index - 1] = clamp_val
            else:  # random intercept

                # intercept and coefficient offset
                rand_inters[clamp_index - (1 + len(coeffs))] = clamp_val

        # iterate over Newton steps for latent mode at fixed tau
        while max_iters > 0:

            # build the Newton system for the current latent values
            linpred, weight_vect, score_vect, response_param, fixed_precision, cross_precision, random_precision_inv, gradient_rand, reduced_precision, gradient_reduced = self.newton_system(
                fixed_vals, response_vals, rand_vals, inter, coeffs, rand_inters, tau, response_param)

            # Newton step with one coordinate held fixed
            if clamp_index is not None:

                # fixed block size
                fixed_count = fixed_precision.shape[0]

                # fixed coordinate in fixed block
                if clamp_index < fixed_count:

                    # boolean array to select coordinates changed by solve
                    fixed_mask = np.ones(fixed_count, dtype=bool)

                    # fixed coordinate excluded from solve
                    fixed_mask[clamp_index] = False

                    # solve reduced system for remaining fixed coordinates
                    step_fixed = np.zeros(fixed_count, dtype=float)
                    step_fixed[fixed_mask] = np.linalg.solve(reduced_precision[np.ix_(fixed_mask, fixed_mask)],
                                                             gradient_reduced[fixed_mask])

                    # random step from back substitution
                    step_rand = random_precision_inv * (gradient_rand - (cross_precision.T @ step_fixed))

                # fixed coordinate in random block
                else:

                    # random index for fixed coordinate
                    rand_index = clamp_index - fixed_count

                    # boolean array to select random coordinates changed by solve
                    rand_mask = np.ones(len(rand_inters), dtype=bool)

                    # fixed coordinate excluded from solve
                    rand_mask[rand_index] = False

                    # reduced system excluding fixed random coordinate
                    rand_prec_inv = random_precision_inv[rand_index]
                    cross_col = cross_precision[:, rand_index]
                    reduced_sub = reduced_precision + (rand_prec_inv * np.outer(cross_col, cross_col))
                    gradient_sub = gradient_reduced + (rand_prec_inv * cross_col * gradient_rand[rand_index])

                    # fixed step from reduced system
                    step_fixed = np.linalg.solve(reduced_sub, gradient_sub)

                    # random step for remaining random coordinates
                    step_rand = np.zeros(len(rand_inters), dtype=float)
                    step_rand[rand_mask] = random_precision_inv[rand_mask] * (
                                gradient_rand[rand_mask] - (cross_precision[:, rand_mask].T @ step_fixed))

                # apply step to fixed block
                inter += step_fixed[0]
                coeffs += step_fixed[1:]

                # apply step to random block
                rand_inters = rand_inters + step_rand

                # stop rule based on step size
                step_max = float(max(np.max(np.abs(step_fixed)), np.max(np.abs(step_rand))))
                if step_max <= tol:
                    break

                # remaining iteration count decremented
                max_iters -= 1
                continue

            # Newton update vector for intercept and coefficients from the reduced system
            step_fixed = scipy.linalg.cho_solve(
                scipy.linalg.cho_factor(reduced_precision, lower=True, check_finite=False), gradient_reduced,
                check_finite=False)

            # Newton update vector for random intercepts from back substitution
            step_rand = random_precision_inv * (gradient_rand - (cross_precision.T @ step_fixed))

            # update intercept, coefficients and random intercepts together using the joint Newton step
            inter += step_fixed[0]
            coeffs += step_fixed[1:]
            rand_inters = rand_inters + step_rand

            # largest latent update tracked for convergence
            step_max = 0
            step_max = max(step_max, np.max(np.abs(step_rand)))
            step_fixed_max = np.max(np.abs(step_fixed))
            step_max = max(step_fixed_max, step_max)

            # rebuild the gradient at the updated point only when the step is small enough to pass the first convergence test
            if step_max <= tol:

                # rebuild the linear predictor and update the response parameter
                linpred = inter + (fixed_vals @ coeffs) + rand_inters[rand_vals]
                response_param = self.likelihood_model.hyper_update(response_vals, linpred, response_param)

                # rebuild the positive curvature weights used by the Newton system
                weight_vect = self.likelihood_model.weight(response_vals, linpred, response_param)

                # rebuild the likelihood score used by the Newton system
                score_vect = self.likelihood_model.score(response_vals, linpred, response_param)
                gradient_coeffs = fixed_vals.T @ score_vect
                gradient_fixed = np.concatenate(([np.sum(score_vect)], gradient_coeffs))

                # add normal prior score
                gradient_fixed[0] = gradient_fixed[0] - (self.intercept_precision_prior * inter)
                gradient_fixed[1:] = gradient_fixed[1:] - (self.fixed_precision_prior * coeffs)

                # curvature scaled score for stopping rule
                curv_inter = float(np.sum(weight_vect))
                curv_coeffs = np.sum(np.asarray(weight_vect).reshape(-1, 1) * (fixed_vals * fixed_vals), axis=0)
                curv_fixed = np.concatenate(([curv_inter], curv_coeffs))
                grad_scaled = float(np.max(np.abs(gradient_fixed) / curv_fixed))

                # stop only when both the overall step and the curvature-scaled gradient are small
                if grad_scaled <= tol:
                    break

            # remaining iteration count decremented
            max_iters -= 1

        # return only the latent mode when requested
        if mode_only == True:
            return inter, coeffs, rand_inters, None, None, None, None, response_param

        # build covariance outputs at the latent mode
        linpred = inter + (fixed_vals @ coeffs) + rand_inters[rand_vals]
        response_param = self.likelihood_model.hyper_update(response_vals, linpred, response_param)
        var_inter, var_coeffs, var_rand, cov_fixed = self.latent_cov(fixed_vals, response_vals, rand_vals, inter,
                                                                     coeffs, rand_inters, tau, response_param)

        return inter, coeffs, rand_inters, var_inter, var_coeffs, var_rand, cov_fixed, response_param

    # reduces fixed-effect precision using the Schur complement identity
    def reduce_precision(self, fixed_precision, fixed_vals, rand_vals, mean_vect, tau, rand_count):

        # calculate per-group sums needed by the random-intercept block
        mean_sum, _ = self.linear_model.group_summaries(mean_vect, np.zeros_like(mean_vect), rand_vals, int(rand_count))

        # build the random-intercept precision diagonal and its elementwise inverse
        random_precision = self.linear_model.random_precision_diag(mean_sum, tau)
        random_precision_inv = 1 / random_precision

        # build the fixed-random cross precision block
        fixed_design = np.concatenate((np.ones((fixed_vals.shape[0], 1)), fixed_vals), axis=1)
        cross_precision = self.linear_model.cross_precision_terms(fixed_design, mean_vect, rand_vals, int(rand_count))

        # reduce the fixed-effect precision using the Schur complement identity
        cross_scaled = cross_precision * random_precision_inv
        reduced_precision = fixed_precision - (cross_scaled @ cross_precision.T)

        return random_precision, reduced_precision, cross_precision

    # approximates conditional latent values given one fixed value
    def conditional(self, fixed_mean, rand_mean, fixed_cov, cross_scale, rand_inv, latent_index, latent_val):

        # convert to numpy for speed
        fixed_mean = np.asarray(fixed_mean)
        rand_mean = np.asarray(rand_mean)
        fixed_cov = np.asarray(fixed_cov)
        cross_scale = np.asarray(cross_scale)
        rand_inv = np.asarray(rand_inv)

        # set sizes for fixed and random blocks
        fixed_count = fixed_mean.shape[0]

        # build a full latent mean vector for updates
        latent_mean = np.concatenate((fixed_mean, rand_mean))
        latent_star = latent_mean.copy()

        # conditional latent value for a fixed effect
        if latent_index < fixed_count:
            # select the fixed effect variance
            fixed_var = float(fixed_cov[latent_index, latent_index])

            # select the fixed cross terms for the selected effect
            fixed_cross = fixed_cov[:, latent_index].copy()
            fixed_cross[latent_index] = 0.0

            # compute the conditional shift for the fixed effect
            fixed_shift = (latent_val - float(fixed_mean[latent_index])) / fixed_var

            # update fixed effects using the Gaussian conditional expectation
            latent_star[:fixed_count] = fixed_mean + (fixed_cross * fixed_shift)

            # update random effects using the Gaussian conditional expectation
            rand_cross = -(fixed_cov[latent_index, :] @ cross_scale)
            latent_star[fixed_count:] = rand_mean + (rand_cross * fixed_shift)

            # overwrite the selected fixed effect
            latent_star[latent_index] = latent_val

            return latent_star

        # convert the full index into a random block index
        rand_index = latent_index - fixed_count

        # build the fixed-random cross covariance column
        fixed_cross = -(fixed_cov @ cross_scale[:, rand_index])

        # build the random covariance column for the selected effect
        rand_col = cross_scale.T @ (fixed_cov @ cross_scale[:, rand_index])
        rand_col[rand_index] = rand_col[rand_index] + rand_inv[rand_index]

        # select the random effect variance
        rand_var = float(rand_col[rand_index])

        # compute the conditional shift for the random effect
        rand_shift = (latent_val - float(rand_mean[rand_index])) / rand_var

        # update fixed effects using the Gaussian conditional expectation
        latent_star[:fixed_count] = fixed_mean + (fixed_cross * rand_shift)

        # update random effects using the Gaussian conditional expectation
        rand_cross = rand_col.copy()
        rand_cross[rand_index] = 0.0
        latent_star[fixed_count:] = rand_mean + (rand_cross * rand_shift)

        # overwrite the selected random effect
        latent_star[latent_index] = latent_val

        return latent_star

    # Laplace curvature penalty to rescale the contribution at the mode by how sharply peaked the integrand is
    def laplace_penalty(self, fixed_vals, response_vals, rand_vals, inter, coeffs, rand_inters, tau, latent_index=None,
                        response_param=None):

        # rebuild linear predictor using final parameters
        linpred = inter + (fixed_vals @ coeffs) + rand_inters[rand_vals]

        # update the response parameter at the current linear predictor
        response_param = self.likelihood_model.hyper_update(response_vals, linpred, response_param)

        # build the positive curvature weights used by the Laplace correction
        mean_vect = self.likelihood_model.weight(response_vals, linpred, response_param)

        # cross curvature and coefficients
        hess_inter = -np.sum(mean_vect)
        hess_inter_coeffs = -(fixed_vals.T @ mean_vect)
        hess_coeffs = -(fixed_vals.T @ (np.asarray(mean_vect).reshape(-1, 1) * fixed_vals))

        # reshape cross curvature terms to fill block matrix
        hess_fixed = np.empty((1 + hess_coeffs.shape[0], 1 + hess_coeffs.shape[1]))
        hess_fixed[0, 0] = hess_inter
        hess_fixed[0, 1:] = hess_inter_coeffs
        hess_fixed[1:, 0] = hess_inter_coeffs
        hess_fixed[1:, 1:] = hess_coeffs

        # build the fixed-effect precision block
        fixed_precision = -hess_fixed

        # add the normal prior precision to the fixed-effect precision
        prior_diag = np.concatenate(
            ([self.intercept_precision_prior], self.fixed_precision_prior * np.ones(fixed_precision.shape[0] - 1)))
        fixed_precision = fixed_precision + np.diag(prior_diag)

        # reduce the fixed-effect precision using the Schur complement identity
        random_precision, reduced_precision, cross_precision = self.reduce_precision(fixed_precision, fixed_vals,
                                                                                     rand_vals, mean_vect, tau,
                                                                                     len(rand_inters))

        # compute the random precision inverse for selective integration
        random_precision_inv = 1 / random_precision

        # log determinant terms needed for the curvature penalty
        random_determ_full = np.sum(np.log(random_precision))

        # factorise the reduced precision using pivoted LU
        lu_full, piv_full = scipy.linalg.lu_factor(reduced_precision, check_finite=False)

        # take the diagonal entries from the packed LU factors
        diag_full = np.diag(lu_full)

        # estimate the permutation parity from the pivot vector
        swap_count_full = np.sum(piv_full != np.arange(piv_full.shape[0]))

        # combine the diagonal polarity with the permutation polarity for the determinant polarity
        reduced_sign_full = np.prod(np.sign(diag_full)) * (-1.0 if (swap_count_full % 2) == 1 else 1.0)

        # sum log magnitudes of the diagonal to get the log determinant magnitude
        reduced_determ_full = np.sum(np.log(np.abs(diag_full)))

        # skip tau values with a non-positive determinant polarity
        if reduced_sign_full <= 0:
            return -np.inf

        # curvature penalty term for tau weighting
        if latent_index is None:
            penalty = -0.5 * (random_determ_full + reduced_determ_full)
            return penalty

        # compute the log determinant after fixing one effect
        fixed_count = fixed_precision.shape[0]
        if latent_index < fixed_count:

            # select all fixed effects except the fixed one
            fixed_mask = np.ones(fixed_count, dtype=bool)
            fixed_mask[int(latent_index)] = False

            # build reduced precision terms with the selected fixed effect removed
            fixed_sub = fixed_precision[np.ix_(fixed_mask, fixed_mask)]
            cross_sub = cross_precision[fixed_mask, :]
            cross_scaled = cross_sub * random_precision_inv
            reduced_sub = fixed_sub - (cross_scaled @ cross_sub.T)

            # factorise the reduced precision using pivoted LU for determinant terms
            lu_drop, piv_drop = scipy.linalg.lu_factor(reduced_sub, check_finite=False)

            # take the diagonal entries from the packed LU factors
            diag_drop = np.diag(lu_drop)

            # estimate the permutation parity from the pivot vector
            swap_count_drop = np.sum(piv_drop != np.arange(piv_drop.shape[0]))

            # combine the diagonal polarity with the permutation polarity for the determinant polarity
            reduced_sign_drop = np.prod(np.sign(diag_drop)) * (-1.0 if (swap_count_drop % 2) == 1 else 1.0)

            # sum log magnitudes of the diagonal to get the log determinant magnitude
            reduced_determ_drop = np.sum(np.log(np.abs(diag_drop)))

            # skip points with an undefined log determinant
            if reduced_sign_drop <= 0:
                return -np.inf

            # curvature penalty term for marginal evaluation
            penalty = -0.5 * (random_determ_full + reduced_determ_drop)
            return penalty

        # convert the full index into a random index
        rand_index = int(latent_index - fixed_count)

        # select all random effects except the fixed one
        rand_mask = np.ones(len(random_precision), dtype=bool)
        rand_mask[rand_index] = False

        # build reduced precision terms with the selected random effect removed
        random_sub = random_precision[rand_mask]
        random_sub_inv = random_precision_inv[rand_mask]
        cross_sub = cross_precision[:, rand_mask]
        cross_scaled = cross_sub * random_sub_inv
        reduced_sub = fixed_precision - (cross_scaled @ cross_sub.T)

        # compute log determinant terms for the curvature correction
        random_determ_drop = np.sum(np.log(random_sub))

        # factorise the reduced precision using pivoted LU for determinant terms
        lu_drop, piv_drop = scipy.linalg.lu_factor(reduced_sub, check_finite=False)

        # take the diagonal entries from the packed LU factors
        diag_drop = np.diag(lu_drop)

        # estimate the permutation parity from the pivot vector
        swap_count_drop = np.sum(piv_drop != np.arange(piv_drop.shape[0]))

        # combine the diagonal polarity with the permutation polarity for the determinant polarity
        reduced_sign_drop = np.prod(np.sign(diag_drop)) * (-1.0 if (swap_count_drop % 2) == 1 else 1.0)

        # sum log magnitudes of the diagonal to get the log determinant magnitude
        reduced_determ_drop = np.sum(np.log(np.abs(diag_drop)))

        # skip points with an undefined log determinant
        if reduced_sign_drop <= 0:
            return -np.inf

        # curvature penalty term for marginal evaluation
        penalty = -0.5 * (random_determ_drop + reduced_determ_drop)
        return penalty

    # estimates the curvature (2nd deriv) of the score around the best precision candidate using three evaluation points
    def hessian(self, fixed_vals, response_vals, rand_vals, best_precision, log_fact):

        # set the curviture of the search
        multiplier = np.exp(probe_step)

        # define a smaller and larger candidate precision values
        smaller_candidate = best_precision / multiplier
        larger_candidate = best_precision * multiplier

        # optimise all three precision values - for a "cold" start (slightly more accurate but slower - activate this block and deactivate the warm start block below)
        # inter_centre, coeffs_centre, rand_centre = optimise_latent(fixed_vals, response_vals, rand_vals, best_precision, latent_tol, max_iters, mode_only=True)[0:3]
        # inter_smaller, coeffs_smaller, rand_smaller = optimise_latent(fixed_vals, response_vals, rand_vals, smaller_candidate, latent_tol, max_iters, mode_only=True)[0:3]
        # inter_larger, coeffs_larger, rand_larger = optimise_latent(fixed_vals, response_vals, rand_vals, larger_candidate, latent_tol, max_iters, mode_only=True)[0:3]

        # warm start block: optimise the centre precision value once and then the nearby precision values using the centre mode
        inter_centre, coeffs_centre, rand_centre = self.optimise_latent(fixed_vals, response_vals, rand_vals,
                                                                        best_precision, latent_tol, max_iters,
                                                                        mode_only=True)[0:3]
        inter_smaller, coeffs_smaller, rand_smaller = self.optimise_latent(fixed_vals, response_vals, rand_vals,
                                                                           smaller_candidate, latent_tol, max_iters,
                                                                           inter_start=inter_centre,
                                                                           coeffs_start=coeffs_centre,
                                                                           rand_inters_start=rand_centre,
                                                                           mode_only=True)[0:3]
        inter_larger, coeffs_larger, rand_larger = self.optimise_latent(fixed_vals, response_vals, rand_vals,
                                                                        larger_candidate, latent_tol, max_iters,
                                                                        inter_start=inter_centre,
                                                                        coeffs_start=coeffs_centre,
                                                                        rand_inters_start=rand_centre, mode_only=True)[0:3]

        # score all three precision values
        score_centre = self.linear_model.tau_logprior(best_precision) + self.tau_upd(fixed_vals, response_vals,
                                                                                     rand_vals, inter_centre,
                                                                                     coeffs_centre, rand_centre,
                                                                                     best_precision, log_fact)
        score_smaller = self.linear_model.tau_logprior(smaller_candidate) + self.tau_upd(fixed_vals, response_vals,
                                                                                         rand_vals, inter_smaller,
                                                                                         coeffs_smaller, rand_smaller,
                                                                                         smaller_candidate, log_fact)
        score_larger = self.linear_model.tau_logprior(larger_candidate) + self.tau_upd(fixed_vals, response_vals,
                                                                                       rand_vals, inter_larger,
                                                                                       coeffs_larger, rand_larger,
                                                                                       larger_candidate, log_fact)

        # add the Laplace penalties
        score_centre += self.laplace_penalty(fixed_vals, response_vals, rand_vals, inter_centre, coeffs_centre, rand_centre,
                                             best_precision)
        score_smaller += self.laplace_penalty(fixed_vals, response_vals, rand_vals, inter_smaller, coeffs_smaller, rand_smaller,
                                              smaller_candidate)
        score_larger += self.laplace_penalty(fixed_vals, response_vals, rand_vals, inter_larger, coeffs_larger, rand_larger,
                                             larger_candidate)

        # convert into log scale
        score_centre = score_centre + np.log(best_precision)
        score_smaller = score_smaller + np.log(smaller_candidate)
        score_larger = score_larger + np.log(larger_candidate)
        step_size = np.log(multiplier)

        # Apply the central second-difference estimator of the second derivative on the log scale
        hessian = (score_larger - (2 * score_centre) + score_smaller) / (step_size * step_size)
        return hessian

    # outer optimiser that searches for the mode of the approximate log posterior in log precision space
    def mode_tau(self, fixed_vals, response_vals, rand_vals, log_fact):

        # track the current mode candidate tau
        tau_mode = self.linear_model.tau_initial()
        log_tau_mode = np.log(tau_mode)

        # optimise the latent field at the starting tau
        inter_mode, coeffs_mode, rand_inters_mode = self.optimise_latent(fixed_vals, response_vals, rand_vals, tau_mode,
                                                                         latent_tol, latent_max_iters, mode_only=True)[0:3]

        # score the starting tau using its conditional latent mode
        score_mode = self.linear_model.tau_logprior(tau_mode) + self.tau_upd(fixed_vals, response_vals, rand_vals,
                                                                             inter_mode, coeffs_mode, rand_inters_mode,
                                                                             tau_mode, log_fact)
        score_mode += self.laplace_penalty(fixed_vals, response_vals, rand_vals, inter_mode, coeffs_mode, rand_inters_mode, tau_mode)

        reuse_side = None
        reuse_log_tau = None
        reuse_score = None
        reuse_inter = None
        reuse_coeffs = None
        reuse_rand = None
        hessian_val = None

        with tqdm(desc="Solving", unit="steps") as pbar:
            for i in range(max_iters):
                # for i in (range(max_iters)):

                # move one step up on the log tau scale
                log_tau_plus = log_tau_mode + mode_step
                tau_plus = np.exp(log_tau_plus)

                # optimise and score the plus candidate
                if (reuse_side == "plus") and np.isclose(log_tau_plus, reuse_log_tau, rtol=0.0, atol=1.0e-12):
                    inter_plus = reuse_inter
                    coeffs_plus = reuse_coeffs
                    rand_inters_plus = reuse_rand
                    score_plus = reuse_score
                else:
                    inter_plus, coeffs_plus, rand_inters_plus = self.optimise_latent(fixed_vals, response_vals, rand_vals,
                                                                                     tau_plus, latent_tol, latent_max_iters,
                                                                                     mode_only=True)[0:3]
                    score_plus = self.linear_model.tau_logprior(tau_plus) + self.tau_upd(fixed_vals, response_vals, rand_vals,
                                                                                         inter_plus, coeffs_plus,
                                                                                         rand_inters_plus, tau_plus, log_fact)
                    score_plus += self.laplace_penalty(fixed_vals, response_vals, rand_vals, inter_plus, coeffs_plus, rand_inters_plus,
                                                       tau_plus)

                # propose one step down on the log tau scale
                log_tau_minus = log_tau_mode - mode_step
                tau_minus = np.exp(log_tau_minus)

                # optimise and score the minus candidate
                if (reuse_side == "minus") and np.isclose(log_tau_minus, reuse_log_tau, rtol=0.0, atol=1.0e-12):
                    inter_minus = reuse_inter
                    coeffs_minus = reuse_coeffs
                    rand_inters_minus = reuse_rand
                    score_minus = reuse_score
                else:
                    inter_minus, coeffs_minus, rand_inters_minus = self.optimise_latent(fixed_vals, response_vals, rand_vals,
                                                                                        tau_minus, latent_tol, latent_max_iters,
                                                                                        mode_only=True)[0:3]
                    score_minus = self.linear_model.tau_logprior(tau_minus) + self.tau_upd(fixed_vals, response_vals, rand_vals,
                                                                                           inter_minus, coeffs_minus,
                                                                                           rand_inters_minus, tau_minus,
                                                                                           log_fact)
                    score_minus += self.laplace_penalty(fixed_vals, response_vals, rand_vals, inter_minus, coeffs_minus, rand_inters_minus,
                                                        tau_minus)

                # stop when neither direction improves the score
                if ((score_plus + log_tau_plus) <= (score_mode + log_tau_mode)) and (
                        (score_minus + log_tau_minus) <= (score_mode + log_tau_mode)):
                    score_mode_log = score_mode + log_tau_mode
                    score_plus_log = score_plus + log_tau_plus
                    score_minus_log = score_minus + log_tau_minus
                    hessian_val = (score_plus_log - (2 * score_mode_log) + score_minus_log) / (mode_step * mode_step)
                    break

                # move to the better direction and carry its latent mode along
                if (score_plus + log_tau_plus) > (score_minus + log_tau_minus):
                    prev_log_tau = log_tau_mode
                    prev_score = score_mode
                    prev_inter = inter_mode
                    prev_coeffs = coeffs_mode
                    prev_rand = rand_inters_mode
                    log_tau_mode = log_tau_plus
                    tau_mode = tau_plus
                    score_mode = score_plus
                    inter_mode = inter_plus
                    coeffs_mode = coeffs_plus
                    rand_inters_mode = rand_inters_plus
                    reuse_side = "minus"
                    reuse_log_tau = prev_log_tau
                    reuse_score = prev_score
                    reuse_inter = prev_inter
                    reuse_coeffs = prev_coeffs
                    reuse_rand = prev_rand
                else:
                    prev_log_tau = log_tau_mode
                    prev_score = score_mode
                    prev_inter = inter_mode
                    prev_coeffs = coeffs_mode
                    prev_rand = rand_inters_mode
                    log_tau_mode = log_tau_minus
                    tau_mode = tau_minus
                    score_mode = score_minus
                    inter_mode = inter_minus
                    coeffs_mode = coeffs_minus
                    rand_inters_mode = rand_inters_minus
                    reuse_side = "plus"
                    reuse_log_tau = prev_log_tau
                    reuse_score = prev_score
                    reuse_inter = prev_inter
                    reuse_coeffs = prev_coeffs
                    reuse_rand = prev_rand

                pbar.update(1)

        # estimate curvature at the mode when the search stopped before bracketing it
        if hessian_val is None:
            hessian_val = self.hessian(fixed_vals, response_vals, rand_vals, tau_mode, log_fact)

        return tau_mode, log_tau_mode, score_mode, hessian_val, inter_mode, coeffs_mode, rand_inters_mode

    # integrates the hyperparameter log posterior curve to produce normalised grid weights
    def integrate_curve(self, tau_candidates, score_vals, grid_step):

        # collect the log posterior scores for weighting on the log tau scale
        log_tau = np.log(np.asarray(tau_candidates, dtype=float))
        score = np.asarray(score_vals, dtype=float)

        # sort points so the weighting is well-defined on an increasing log tau grid
        sort_index = np.argsort(log_tau)
        log_tau = log_tau[sort_index]
        score = score[sort_index]

        # stabilise exponentiation for numerical weighting on the log tau scale
        log_dens = score + log_tau
        log_shift = float(np.max(log_dens))
        weight_raw = np.exp(log_dens - log_shift)

        # apply trapezoid area weights from the regular grid spacing
        weight_raw = weight_raw * grid_step
        weight_raw[0] = 0.5 * weight_raw[0]
        weight_raw[-1] = 0.5 * weight_raw[-1]

        # normalise the allocated weights to sum to one
        weight_sorted = weight_raw / float(np.sum(weight_raw))

        # map weights back to the original candidate order
        tau_weights = [0.0] * len(tau_candidates)
        for index_pos in range(len(sort_index)):
            tau_weights[int(sort_index[index_pos])] = float(weight_sorted[index_pos])

        return tau_weights

    # searches a grid around the mode and returns tau candidates, weights and latent modes
    def grid_search(self, fixed_vals, response_vals, rand_vals, log_fact):

        tau_candidates = []
        latent_modes = []
        score_vals = []

        # find the mode of the hyperparameter score and get curvature at that mode
        tau_mode, log_tau_mode, score_mode, hessian_val, inter_mode, coeffs_mode, rand_inters_mode = self.mode_tau(
            fixed_vals, response_vals, rand_vals, log_fact)
        tau_candidates.append(tau_mode)
        latent_modes.append((inter_mode, coeffs_mode, rand_inters_mode))
        score_vals.append(score_mode)

        # flip sign and take inverse square root of curvature for grid spacing
        curvature = -hessian_val
        if (not np.isfinite(curvature)) or (curvature <= 0.0):
            grid_step = mode_step
        else:
            grid_step = grid_dz / math.sqrt(curvature)

        # expand symmetrically around the mode on a fixed grid and truncate afterwards using stop
        for i in range(1, grid_max + 1):
            # plus side candidate optimising and scoring
            log_tau_plus = log_tau_mode + (i * grid_step)
            tau_plus = np.exp(log_tau_plus)
            inter_plus, coeffs_plus, rand_inters_plus = self.optimise_latent(fixed_vals, response_vals, rand_vals,
                                                                             tau_plus, latent_tol, latent_max_iters,
                                                                             mode_only=True)[0:3]
            score_plus = self.linear_model.tau_logprior(tau_plus) + self.tau_upd(fixed_vals, response_vals, rand_vals,
                                                                                 inter_plus, coeffs_plus,
                                                                                 rand_inters_plus, tau_plus, log_fact)
            score_plus += self.laplace_penalty(fixed_vals, response_vals, rand_vals, inter_plus, coeffs_plus, rand_inters_plus,
                                               tau_plus)

            # update tau candidates and latent mode
            tau_candidates.append(tau_plus)
            latent_modes.append((inter_plus, coeffs_plus, rand_inters_plus))
            score_vals.append(score_plus)

            # minus side candidate optimising and scoring
            log_tau_minus = log_tau_mode - (i * grid_step)
            tau_minus = np.exp(log_tau_minus)
            inter_minus, coeffs_minus, rand_inters_minus = self.optimise_latent(fixed_vals, response_vals, rand_vals,
                                                                                tau_minus, latent_tol, latent_max_iters,
                                                                                mode_only=True)[0:3]
            score_minus = self.linear_model.tau_logprior(tau_minus) + self.tau_upd(fixed_vals, response_vals, rand_vals,
                                                                                   inter_minus, coeffs_minus,
                                                                                   rand_inters_minus, tau_minus,
                                                                                   log_fact)
            score_minus += self.laplace_penalty(fixed_vals, response_vals, rand_vals, inter_minus, coeffs_minus, rand_inters_minus,
                                                tau_minus)

            # update tau candidates and latent mode
            tau_candidates.append(tau_minus)
            latent_modes.append((inter_minus, coeffs_minus, rand_inters_minus))
            score_vals.append(score_minus)

        # compute log posterior density for each precision candidate on the log scale
        log_tau_all = np.log(np.asarray(tau_candidates))
        log_dens_all = np.asarray(score_vals) + log_tau_all

        # keep only candidates within stop log density units of the peak
        log_peak = np.max(log_dens_all)
        keep_mask = (log_dens_all >= (log_peak - stop))

        # filter candidates and aligned latent modes and scores using the keep mask
        tau_candidates = [tau_candidates[i] for i in range(len(tau_candidates)) if bool(keep_mask[i])]
        latent_modes = [latent_modes[i] for i in range(len(latent_modes)) if bool(keep_mask[i])]
        score_vals = [score_vals[i] for i in range(len(score_vals)) if bool(keep_mask[i])]

        # integrate the interpolated hyperparameter log posterior to obtain normalised tau weights
        tau_weights = self.integrate_curve(tau_candidates, score_vals, grid_step)

        return tau_candidates, tau_weights, latent_modes

    # returns mean and variance after averaging conditional Gaussians
    def tau_meanvar(self, means_by_tau, vars_by_tau, tau_weights):
        mix_mean = 0.0
        mix_second = 0.0

        # accumulate weighted mean and weighted second moment across tau
        for i in range(len(tau_weights)):
            mix_mean += tau_weights[i] * means_by_tau[i]  # mean
            mix_second += tau_weights[i] * (vars_by_tau[i] + means_by_tau[i] ** 2)  # second moment

        # compute variance from second moment and mean
        mix_var = mix_second - mix_mean * mix_mean

        return mix_mean, mix_var

    # unnormalised log marginals at Gauss-Hermite evaluation points
    def log_marginals(self, fixed_vals, response_vals, rand_vals, tau, fixed_mean, rand_inters, fixed_cov, cross_scale,
                      rand_inv, eval_points,
                      fixed_count, rand_count, latent_offset, log_fact):

        # compute unnormalised log marginals at the evaluation points
        marginals = np.empty_like(eval_points)

        for i in range(marginals.shape[0]):
            # set the latent index for this row once
            latent_index = latent_offset + i

            # compute Laplace denominator correction once per latent component (centre node)
            centre_index = int(eval_points.shape[1] // 2)
            centre_val = eval_points[i, centre_index]
            centre_star = self.conditional(fixed_mean, rand_inters, fixed_cov, cross_scale, rand_inv, latent_index,
                                           centre_val)
            inter_centre = centre_star[0]
            coeffs_centre = centre_star[1:fixed_count]
            rand_centre = centre_star[fixed_count:]
            penalty_cache = self.laplace_penalty(fixed_vals, response_vals, rand_vals, inter_centre, coeffs_centre, rand_centre, tau,
                                                 latent_index=latent_index)

            for j in range(marginals.shape[1]):
                # build implied latent effects for this evaluation point
                latent_val = eval_points[i, j]
                latent_star = self.conditional(fixed_mean, rand_inters, fixed_cov, cross_scale, rand_inv, latent_index,
                                               latent_val)

                # extract implied intercept, coefficients and random intercepts
                inter_star = latent_star[0]
                coeffs_star = latent_star[1:fixed_count]
                rand_star = latent_star[fixed_count:]

                # unnormalised log marginal for this evaluation point
                marginals[i, j] = self.tau_upd(fixed_vals, response_vals, rand_vals, inter_star, coeffs_star, rand_star,
                                               tau, log_fact)

                # add the tau hyperprior term
                marginals[i, j] += self.linear_model.tau_logprior(tau)

                # add the log-scale jacobian for integrating on log tau
                marginals[i, j] += np.log(tau)

                # Laplace denominator correction term reused across GH nodes
                marginals[i, j] += penalty_cache

        return marginals

    # corrected marginal means and variances from unnormalised log marginals
    def corrected_marginals(self, marginals, eval_points, base_means, base_sds, gh_std_nodes, gh_std_weights,
                            extrapolate=0.0, factor=15):

        # precompute constants used by the Gaussian log density
        log_two_pi = np.log(2.0 * np.pi)

        # compute the Gaussian base log marginal at the evaluation points
        centred = (eval_points - base_means.reshape(-1, 1)) / base_sds.reshape(-1, 1)
        gauss_log = (-0.5 * (centred * centred)) - np.log(base_sds.reshape(-1, 1)) - (0.5 * log_two_pi)

        # compute the log differences used for the correction fit
        log_diff = marginals - gauss_log

        # allocate corrected moments
        mean_corr = np.empty(eval_points.shape[0], dtype=float)
        var_corr = np.empty(eval_points.shape[0], dtype=float)

        for i in range(eval_points.shape[0]):

            # drop non-finite support values caused by bounded priors returning minus infinity
            finite_mask = np.isfinite(log_diff[i, :])

            # fall back to the Gaussian marginal when there are not enough finite support points
            if np.sum(finite_mask) < 4:
                mean_corr[i] = float(base_means[i])
                var_corr[i] = float(base_sds[i] * base_sds[i])
                continue

            # restrict the correction fit to the finite Gauss-Hermite support
            eval_x = eval_points[i, finite_mask]
            eval_y = log_diff[i, finite_mask]

            # build integration support from the finite evaluation support
            range_val = float(eval_x[-1] - eval_x[0])
            grid_lower = float(eval_x[0] - (extrapolate * range_val))
            grid_upper = float(eval_x[-1] + (extrapolate * range_val))

            # build the x-grid used to represent the corrected marginal density
            grid_count = int(factor * eval_x.shape[0])
            if (grid_count % 2) == 0:
                grid_count = grid_count - 1
            grid_vals = np.linspace(grid_lower, grid_upper, grid_count)

            # fit the log-density ratio spline on the finite evaluation support
            spline = scipy.interpolate.CubicSpline(eval_x, eval_y, bc_type="natural")

            # build the corrected log density on the x-grid
            centred_grid = (grid_vals - base_means[i]) / base_sds[i]
            log_gauss = (-0.5 * (centred_grid * centred_grid)) - np.log(base_sds[i]) - (0.5 * log_two_pi)
            log_corr = log_gauss + spline(grid_vals)

            # build an explicit marginal object on the x-grid
            log_shift = float(np.max(log_corr))
            dens_vals = np.exp(log_corr - log_shift)
            marginal = {"x": grid_vals, "y": dens_vals}

            # normalise and integrate moments under the corrected marginal using Simpson's rule
            norm = scipy.integrate.simpson(marginal["y"], marginal["x"])
            mean_val = scipy.integrate.simpson(marginal["x"] * marginal["y"], marginal["x"]) / norm
            second_val = scipy.integrate.simpson(marginal["x"] * marginal["x"] * marginal["y"], marginal["x"]) / norm

            # corrected marginal mean and variance
            mean_corr[i] = float(mean_val)
            var_corr[i] = float(second_val - (mean_val * mean_val))

        return mean_corr, var_corr

    # integrated nested Laplace approximation (INLA) as the variance estimator
    def inla(self, fixed_vals, response_vals, rand_vals):

        # coerce the model inputs to numpy once so the optimiser path is numpy-only
        fixed_vals = fixed_vals.to_numpy()
        response_vals = response_vals.to_numpy()
        rand_vals = rand_vals.to_numpy()

        # death by list, you say? allow me to oblige!
        inter_means = []
        inter_vars = []
        coeffs_means = []
        coeffs_vars = []
        rand_means = []
        rand_vars = []
        posterior_fixedmeans = []
        posterior_fixedvars = []
        posterior_randmeans = []
        posterior_randvars = []
        posterior_fixedcov = []
        fixed_covs = []
        fixed_gh = []
        fixed_mix = []
        rand_mix = []

        # Gauss-Hermite nodes and weights from the Hermite polynomial
        gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(gh_count)

        # rescale the abscissae and weights so they correspond to the standard normal distribution
        gh_std_nodes = np.sqrt(2.0) * gh_nodes
        gh_std_weights = gh_weights / np.sqrt(np.pi)

        # precompute response-only constants for the likelihood
        log_fact = self.likelihood_model.precompute(response_vals, self.response_param_start)

        # build the tau grid so inla can integrate over tau
        tau_candidates, tau_weights, latent_modes = self.grid_search(fixed_vals, response_vals, rand_vals, log_fact)

        # store response parameters for each tau candidate
        response_params = []

        # compute conditional variances per grid point
        for i in tqdm(range(len(tau_candidates)), desc="Solving", unit="step"):
            # compute conditional variances at the stored latent mode for this tau
            tau_val = tau_candidates[i]
            inter, coeffs, rand_inters = latent_modes[i]
            linpred = inter + (fixed_vals @ coeffs) + rand_inters[rand_vals]
            response_param = self.likelihood_model.hyper_update(response_vals, linpred, self.response_param_start)
            var_inter, var_coeffs, var_rand, cov_fixed = self.latent_cov(fixed_vals, response_vals, rand_vals, inter,
                                                                         coeffs, rand_inters, tau_val, response_param)

            # store the data
            fixed_covs.append(cov_fixed)

            # store Gaussian random summaries without marginal correction
            rand_means.append(rand_inters.copy())
            rand_vars.append(var_rand.copy())
            response_params.append(response_param)

            # align Gauss-Hermite quadrature with fixed effects
            fixed_mean = np.concatenate(([inter], coeffs))
            fixed_sd = np.sqrt(np.concatenate(([var_inter], var_coeffs)))
            fixed_eval = fixed_mean.reshape(-1, 1) + (fixed_sd.reshape(-1, 1) * gh_std_nodes.reshape(1, -1))
            fixed_gh.append(fixed_eval)

            # build the positive curvature weights on the same observation shape as the response
            mean_vect = self.likelihood_model.weight(response_vals, linpred, response_param)

            # compute grouped means and grouped score sums
            mean_sum, _ = self.linear_model.group_summaries(mean_vect, np.zeros_like(mean_vect), rand_vals,
                                                            len(rand_inters))

            # compute the random precision diagonal and its inverse
            random_precision = self.linear_model.random_precision_diag(mean_sum, tau_val)
            random_precision_inv = 1 / random_precision

            # compute cross_precision
            fixed_design = np.concatenate((np.ones((fixed_vals.shape[0], 1)), fixed_vals), axis=1)
            cross_precision = self.linear_model.cross_precision_terms(fixed_design, mean_vect, rand_vals,
                                                                      len(rand_inters))

            # store values for conditional
            cross_scale = cross_precision * random_precision_inv
            rand_inv = random_precision_inv

            # grouped sums for draw functions
            cross_cov = -(cov_fixed @ cross_scale)

            # store random intercept means, variances and cross_cov for this tau
            fixed_mix.append((np.concatenate(([inter], coeffs)), cov_fixed))
            rand_mix.append((rand_inters, var_rand, cross_cov))

            # set sizes for fixed and random terms
            fixed_count = cov_fixed.shape[0]
            rand_count = len(rand_inters)

            # unnormalised log marginals for fixed effects
            fixed_la = self.log_marginals(fixed_vals, response_vals, rand_vals, tau_val, fixed_mean, rand_inters,
                                          cov_fixed, cross_scale, rand_inv,
                                          fixed_eval, fixed_count, rand_count, 0, log_fact)

            # overwrite per-tau Gaussian fixed summaries with corrected summaries
            fixed_corrmeans, fixed_corrvars = self.corrected_marginals(fixed_la, fixed_eval, fixed_mean, fixed_sd,
                                                                       gh_std_nodes, gh_std_weights)
            inter_means.append(fixed_corrmeans[0])
            inter_vars.append(fixed_corrvars[0])
            coeffs_means.append(fixed_corrmeans[1:])
            coeffs_vars.append(fixed_corrvars[1:])

        # integrate the intercept marginal over tau
        posterior_intermean, posterior_intervar = self.tau_meanvar(inter_means, inter_vars, tau_weights)

        # fixed effect mean and variance loop after averaging across tau
        coeffs_means_np = np.vstack(coeffs_means)
        coeffs_vars_np = np.vstack(coeffs_vars)
        tau_weights_np = np.asarray(tau_weights)
        posterior_fixedmeans_np = tau_weights_np @ coeffs_means_np
        posterior_fixedsecond_np = tau_weights_np @ (coeffs_vars_np + (coeffs_means_np * coeffs_means_np))
        posterior_fixedvars_np = posterior_fixedsecond_np - (posterior_fixedmeans_np * posterior_fixedmeans_np)
        posterior_fixedmeans = list(posterior_fixedmeans_np)
        posterior_fixedvars = list(posterior_fixedvars_np)

        # rebuild the fixed-effect mean vector for covariance integration
        posterior_fixedmean = np.concatenate(([posterior_intermean], np.asarray(posterior_fixedmeans)))

        # initialise the fixed-effect covariance accumulator
        posterior_fixedsecond = np.zeros_like(fixed_covs[0])

        # accumulate the fixed-effect covariance terms across tau
        for i in range(len(tau_weights)):
            mean_tau = np.concatenate(([inter_means[i]], coeffs_means[i]))
            posterior_fixedsecond += tau_weights[i] * (fixed_covs[i] + np.outer(mean_tau, mean_tau))

        # compute the fixed-effect covariance after tau averaging
        posterior_fixedcov = posterior_fixedsecond - np.outer(posterior_fixedmean, posterior_fixedmean)

        # random intercept mean and variance loop after averaging across tau
        rand_means_np = np.vstack(rand_means)
        rand_vars_np = np.vstack(rand_vars)
        tau_weights_np = np.asarray(tau_weights)
        posterior_randmeans_np = tau_weights_np @ rand_means_np
        posterior_randsecond_np = tau_weights_np @ (rand_vars_np + (rand_means_np * rand_means_np))
        posterior_randvars_np = posterior_randsecond_np - (posterior_randmeans_np * posterior_randmeans_np)
        posterior_randmeans = list(posterior_randmeans_np)
        posterior_randvars = list(posterior_randvars_np)

        # returns: intercept posterior mean and variance,
        # fixed-effect posterior means, variances and covariance,
        # random-intercept posterior means and variance
        # posterior random-intercept precision candidates and weights (approximation to the marginal posterior over tau)
        # per-tau fixed-effect mean vectors and covariance matrices for draw-based intervals
        # per-tau random-intercept means, variances and fixed-random cross covariances for draw-based intervals
        return (posterior_intermean, posterior_intervar, posterior_fixedmeans,
                posterior_fixedvars, posterior_fixedcov, posterior_randmeans, posterior_randvars, tau_candidates,
                tau_weights, fixed_mix, rand_mix, response_params)

    # draw joint fixed and random latent effects from the posterior approximation
    def latent_effect_draws(self, posterior_intermean, posterior_fixedmeans, posterior_fixedcov,
                            posterior_randmeans, posterior_randvars,
                            draws_count=1000, fixed_mix=None, rand_mix=None, tau_weights=None,
                            return_tau_index=False):

        # keep track of the sampled tau component when downstream code needs matching likelihood hyperparameters
        tau_index = None

        # convert random-effect summaries to arrays for vectorised draws
        posterior_randmeans = np.asarray(posterior_randmeans)
        posterior_randvars = np.asarray(posterior_randvars)

        # draw joint fixed effects from the averaged Gaussian approximation
        fixed_mean = np.concatenate(([posterior_intermean], np.asarray(posterior_fixedmeans)))
        fixed_draws = np.random.multivariate_normal(fixed_mean, posterior_fixedcov, size=draws_count)

        # draw random intercepts independently from the averaged marginal approximation
        rand_draws = np.random.normal(posterior_randmeans, np.sqrt(posterior_randvars),
                                      size=(draws_count, len(posterior_randmeans)))

        # switch to the per-tau mixture approximation when the conditional summaries are available
        if fixed_mix is not None:

            # sample tau values according to their posterior weights
            tau_index = np.random.choice(len(fixed_mix), size=draws_count, p=np.asarray(tau_weights))

            # draw fixed-effect and random-effect samples from the corresponding conditional posteriors
            fixed_draws = np.empty((draws_count, fixed_mix[0][1].shape[0]))
            rand_draws = np.empty((draws_count, len(rand_mix[0][0])))

            # iterate over posterior draws so each draw uses its own sampled tau component
            for i in range(draws_count):

                # draw the fixed effects from the conditional Gaussian for this tau value
                mean_tau, cov_tau = fixed_mix[int(tau_index[i])]
                fixed_draws[i, :] = np.random.multivariate_normal(mean_tau, cov_tau)

                # recover the conditional random-effect moments implied by the fixed draw
                rand_means_tau, rand_vars_tau, cross_cov_tau = rand_mix[int(tau_index[i])]
                fixed_shift = np.linalg.solve(cov_tau, (fixed_draws[i, :] - mean_tau))
                rand_cov_solve = np.linalg.solve(cov_tau, cross_cov_tau)
                rand_mean = rand_means_tau + (cross_cov_tau.T @ fixed_shift)
                rand_var = rand_vars_tau - np.sum(cross_cov_tau * rand_cov_solve, axis=0)
                
                # draw the random intercepts from that conditional Gaussian
                rand_draws[i, :] = np.random.normal(rand_mean, np.sqrt(rand_var))

        # return the sampled tau component alongside the latent draws when later code needs it
        if return_tau_index:
            return fixed_draws, rand_draws, tau_index

        return fixed_draws, rand_draws

    # expected counts and uncertainty interval (per observation) - to compare model fit against observed counts
    def expected_counts(self, fixed_vals, rand_vals, posterior_intermean, posterior_fixedmeans, posterior_fixedcov,
                        posterior_randmeans, posterior_randvars,
                        draws_count=1000, fixed_mix=None, rand_mix=None, tau_weights=None,
                        batch_size=summary_batch_size):

        # convert inputs to arrays for numpy indexing and reuse the cached fixed-design matrix when possible
        fixed_design = self.fixed_design if fixed_vals is self.fixed_vals else np.concatenate(
            (np.ones((fixed_vals.shape[0], 1)), fixed_vals.to_numpy()), axis=1)
        rand_np = rand_vals.to_numpy()

        # draw posterior latent effects once and reuse them across observation batches
        fixed_draws, rand_draws = self.latent_effect_draws(posterior_intermean, posterior_fixedmeans,
                                                           posterior_fixedcov, posterior_randmeans,
                                                           posterior_randvars, draws_count=draws_count,
                                                           fixed_mix=fixed_mix, rand_mix=rand_mix,
                                                           tau_weights=tau_weights)

        # allocate the fitted summaries per observation
        mean = np.empty(fixed_design.shape[0])
        mean_lower = np.empty(fixed_design.shape[0])
        mean_upper = np.empty(fixed_design.shape[0])

        # compute fitted mean summaries in row batches to avoid materialising all draws at once
        with tqdm(total=fixed_design.shape[0], desc="Fitted mean summaries", unit="row") as summary_bar:
            for batch_start in range(0, fixed_design.shape[0], batch_size):
                batch_stop = min(batch_start + batch_size, fixed_design.shape[0])
                fixed_batch = fixed_design[batch_start:batch_stop, :]
                rand_batch = rand_np[batch_start:batch_stop]
                linpred_batch = (fixed_draws @ fixed_batch.T) + rand_draws[:, rand_batch]
                mean_draws_batch = self.likelihood_model.mean(linpred_batch)

                # compute posterior mean and equal-tailed credible bounds for this batch of observations
                mean[batch_start:batch_stop] = np.mean(mean_draws_batch, axis=0)
                mean_lower[batch_start:batch_stop] = np.quantile(mean_draws_batch, 0.025, axis=0)
                mean_upper[batch_start:batch_stop] = np.quantile(mean_draws_batch, 0.975, axis=0)
                summary_bar.update(batch_stop - batch_start)

        return mean, mean_lower, mean_upper, None

    # simulate the dataset from the posterior predictives by drawing from a normal distribution
    def sim_posterior(self, fixed_vals, response_vals, rand_vals, posterior_intermean, posterior_fixedmeans,
                      posterior_fixedcov, posterior_randmeans, posterior_randvars, draws=100, fixed_mix=None,
                      rand_mix=None, tau_weights=None, response_params=None):

        replicate_outcomes = []
        fixed_vals = fixed_vals.to_numpy()
        response_vals = response_vals.to_numpy()
        rand_vals = rand_vals.to_numpy()
        density = list(response_vals)

        # sample the tau component for each predictive draw when per-tau summaries are supplied
        if fixed_mix is not None:
            # choose which tau value to use for each draw
            tau_index = np.random.choice(len(fixed_mix), size=draws, p=np.asarray(tau_weights))

        # iterate over posterior draws and simulate one replicated dataset from each draw
        for i in range(draws):

            # start from the averaged fixed and random results
            fixed_mean = np.concatenate(([posterior_intermean], np.asarray(posterior_fixedmeans)))
            fixed_cov = posterior_fixedcov
            rand_means = posterior_randmeans
            rand_vars = posterior_randvars

            # if saved per-tau results are provided, use the ones for this draw
            response_param = self.response_param_start
            # replace the averaged summaries by the tau-specific ones for this draw when available
            if fixed_mix is not None:
                fixed_mean, fixed_cov = fixed_mix[int(tau_index[i])]
                rand_means, rand_vars, cross_cov = rand_mix[int(tau_index[i])]
                
                # use the tau-specific response parameter for this draw when it was saved
                if response_params is not None:
                    response_param = response_params[int(tau_index[i])]

            # draw fixed effects from the normal approximation and take the intercept
            fixed_draw = np.random.multivariate_normal(fixed_mean, fixed_cov)
            inter_draw = fixed_draw[0]

            # take the fixed-effect coefficients from the same draw
            coeffs_draws = list(fixed_draw[1:])

            rand_inters = []
            # draw random effects differently depending on whether tau-specific summaries are available
            if fixed_mix is None:
                # draw random intercepts independently when per-tau results are not used

                # iterate over random-effect levels and draw each averaged-posterior intercept
                for j in range(len(rand_means)):
                    rand_inter = np.random.normal(rand_means[j], np.sqrt(rand_vars[j]))
                    rand_inters.append(rand_inter)
            else:
                # use the saved random intercept values for this draw
                rand_means, rand_vars, cross_cov = rand_mix[int(tau_index[i])]

                # compute fixed_cov^{-1} times (fixed_draw - fixed_mean)
                fixed_shift = np.linalg.solve(fixed_cov, (fixed_draw - fixed_mean))

                # iterate over random-effect levels and draw each conditional random intercept
                for j in range(len(rand_means)):

                    # take the column for this random intercept level
                    rand_cov = cross_cov[:, j]

                    # adjust the mean for this random intercept
                    rand_mean = rand_means[j] + float(np.dot(rand_cov, fixed_shift))

                    # adjust the variance for this random intercept
                    rand_var = rand_vars[j] - float(np.dot(rand_cov, np.linalg.solve(fixed_cov, rand_cov)))
                    rand_inter = np.random.normal(rand_mean, np.sqrt(rand_var))
                    rand_inters.append(rand_inter)

            replicates = []
            # iterate over observations and simulate one outcome under the drawn latent state
            for j in range(len(response_vals)):
                linpred = inter_draw  # start with the intercept

                # add the fixed-effect contribution to the linear predictor

                # accumulate the fixed-effect contribution term by term for this observation
                for k in range(len(coeffs_draws)):
                    linpred += (fixed_vals[j, k] * coeffs_draws[k])

                # add the random intercept contribution to the linear predictor
                linpred += rand_inters[rand_vals[j]]

                # draw an outcome from the likelihood conditional on the linear predictor
                outcome = self.likelihood_model.draw(linpred, hyper=response_param)
                replicates.append(outcome)

            # store the replicated dataset for the density overlay
            replicate_outcomes.append(replicates)
            density += replicates

        return density, replicate_outcomes

    """
    Posterior predictive check methods

    """

    # posterior predictive check visualisation (based on Gabry et al. 2019)
    def check(self, fixed_vals, response_vals, rand_vals, posterior_intermean, posterior_intervar,
              posterior_fixedmeans, posterior_fixedvars, posterior_fixedcov, posterior_randmeans, posterior_randvars,
              tau_weights=None, fixed_mix=None, rand_mix=None, response_params=None):

        # set the replicate draw count for the posterior predictive check
        draws_count = 100

        # translate model inputs to numpy arrays and reuse the cached fixed-design matrix when possible
        fixed_design = self.fixed_design if fixed_vals is self.fixed_vals else np.concatenate(
            (np.ones((fixed_vals.shape[0], 1)), fixed_vals.to_numpy()), axis=1)
        response_vals = response_vals.to_numpy()
        rand_vals = rand_vals.to_numpy()

        # draw posterior latent effects once and reuse them across observation batches
        fixed_draws, rand_draws, tau_index = self.latent_effect_draws(
            posterior_intermean, posterior_fixedmeans, posterior_fixedcov, posterior_randmeans, posterior_randvars,
            draws_count=draws_count, fixed_mix=fixed_mix, rand_mix=rand_mix, tau_weights=tau_weights,
            return_tau_index=True)

        # align the response parameter draws with the sampled latent-effect draws when they are available
        response_param_draws = None
        if response_params is not None:
            response_param_draws = np.asarray([response_params[int(tau_index[i])] for i in range(draws_count)])
        
        # reshape the response parameter draws so the likelihood can broadcast them across observation batches
        response_param_batch = None if response_param_draws is None else response_param_draws[:, None]

        # use every posterior predictive draw in the density overlay
        plot_index = np.arange(draws_count)
        replicate_plot_draws = np.empty((draws_count, fixed_design.shape[0]))

        # generate the medians as a test stat
        groups = np.asarray(rand_vals)

        # half-integer bin edges so each group falls into its own bin
        bins = np.arange(-0.5, np.max(groups) + 1.5, 1.0)

        # compute observed group means and allocate replicated group sums
        obs_means, _, _ = scipy.stats.binned_statistic(groups, response_vals, statistic='mean', bins=bins)
        group_counts, _, _ = scipy.stats.binned_statistic(groups, response_vals, statistic='count', bins=bins)
        sim_group_sums = np.zeros((draws_count, len(obs_means)))

        # ensure the density overlay is on the fitted response scale supplied to the model
        response_plot = np.asarray(response_vals)

        # draw posterior predictive outcomes in observation batches to avoid a full draws x rows matrix
        with tqdm(total=fixed_design.shape[0], desc="Posterior predictive", unit="row") as posterior_bar:
            for batch_start in range(0, fixed_design.shape[0], summary_batch_size):
                batch_stop = min(batch_start + summary_batch_size, fixed_design.shape[0])
                fixed_batch = fixed_design[batch_start:batch_stop, :]
                rand_batch = rand_vals[batch_start:batch_stop]
                linpred_batch = (fixed_draws @ fixed_batch.T) + rand_draws[:, rand_batch]
                mean_draws_batch = self.likelihood_model.mean(linpred_batch)

                # draw replicated outcomes under the active likelihood instead of assuming a Poisson model
                if response_param_batch is None:
                    replicate_draws_batch = self.likelihood_model.draw_from_mean(mean_draws_batch)
                else:
                    replicate_draws_batch = self.likelihood_model.draw_from_mean(mean_draws_batch, hyper=response_param_batch)

                # store only the subset of replicated datasets used by the density overlay
                replicate_plot_draws[:, batch_start:batch_stop] = replicate_draws_batch[plot_index, :]

                # accumulate replicated group sums for every draw
                for draw_index in range(draws_count):
                    np.add.at(sim_group_sums[draw_index, :], rand_batch, replicate_draws_batch[draw_index, :])

                # update the progress bar once per completed observation batch
                posterior_bar.update(batch_stop - batch_start)

        # convert replicated group sums into group means
        sim_means = (sim_group_sums / group_counts[np.newaxis, :]).T

        # sample the observed transformed values down to the overlay plotting size
        plot_rng = np.random.default_rng(42)
        response_plot_density = response_plot
        if response_plot.shape[0] > density_sample_size:
            response_plot_density = response_plot[
                plot_rng.choice(response_plot.shape[0], size=density_sample_size, replace=False)]

        # collect the transformed replicated overlay samples before defining the plotting grid
        replicate_plot_sets = []
        for index_pos in range(draws_count):
            replicate_plot = np.asarray(replicate_plot_draws[index_pos, :])

            # sample each replicated dataset down to the overlay plotting size
            if replicate_plot.shape[0] > density_sample_size:
                replicate_plot = replicate_plot[
                    plot_rng.choice(replicate_plot.shape[0], size=density_sample_size, replace=False)]

            # keep replicated values on the same fitted response scale as the observed values
            replicate_plot_sets.append(replicate_plot)

        # build the plotting grid from the pooled overlay samples so extreme draws do not flatten the visible curves
        pooled_plot_vals = np.concatenate([response_plot_density] + replicate_plot_sets)
        grid_lower = float(np.percentile(pooled_plot_vals, 0.5))
        grid_upper = float(np.percentile(pooled_plot_vals, 99.5))
        grid_vals = np.linspace(grid_lower, grid_upper, 500)

        # generate a (smoothed) kernel density estimate for the observed outcome (y axis)
        smooth_density = scipy.stats.gaussian_kde(response_plot_density)(grid_vals)

        # housekeeping
        density_plot_start = time.perf_counter()
        fig1, ax1 = plt.subplots()
        density_list = []

        # draw each simulated replicate
        for replicate_plot in replicate_plot_sets:
            smooth_replicate = scipy.stats.gaussian_kde(replicate_plot)(grid_vals)
            density_list.append(smooth_replicate)
            ax1.plot(grid_vals, smooth_replicate, color='lightgrey', alpha=0.4)

        # draw the replicated median density curve
        density_array = np.vstack(density_list)
        density_mid = np.percentile(density_array, 50.0, axis=0)
        ax1.plot(grid_vals, density_mid, color="grey", linewidth=1.5)

        # plot the observed density overlay
        ax1.plot(grid_vals, smooth_density, "-", color="black", linewidth=2.0)

        # publication version keeps the overlay visually clean
        # ax1.set_title("posterior predictive check")
        # ax1.set_xlabel("Dependent variable")
        # ax1.set_ylabel("Density")
        # ax1.legend()
        # ax1.set_xticks([])
        ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        ax1.set_yticks([])
        fig1.savefig("posterior_predictive_check.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig1)

        # print the density overlay plot runtime
        density_plot_elapsed = time.perf_counter() - density_plot_start
        density_plot_minutes = int((density_plot_elapsed % 3600) // 60)
        density_plot_seconds = int(density_plot_elapsed % 60)
        print(f"Posterior predictive runtime: {density_plot_minutes}:{density_plot_seconds}")

        # plot replicated-median histograms with the observed median as a reference line
        groups = len(obs_means)
        plot_group_count = min(groups, 30)
        cols = 3
        rows = math.ceil(plot_group_count / cols)

        # create a grid of panels for the group-wise median checks
        group_plot_start = time.perf_counter()
        fig, axes = plt.subplots(rows, cols, squeeze=False)
        axes_flat = axes.ravel()

        # draw the replicated median distribution and the observed median for each group
        for i in range(plot_group_count):
            axes_flat[i].hist(sim_means[i, :], bins=30, color='lightgrey', alpha=0.4)
            axes_flat[i].axvline(obs_means[i], linewidth=2.0)
            # axes_flat[i].set_title(f"group {i + 1}")
            # axes_flat[i].set_xlabel("mean")
            # axes_flat[i].set_ylabel("Density")
            # axes_flat[i].set_xticks(np.arange(...))
            # axes_flat[i].set_yticks([])

            # remove subplot ticks in the publication version
            axes_flat[i].set_xticks([])
            axes_flat[i].set_yticks([])

        # hide any unused panels
        for i in range(plot_group_count, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # fig.tight_layout()
        fig.savefig("posterior_predictive_histograms.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return

    # PSIS-LOO helper: empirical Bayes procedure (Zhang and Stephens, 2009)
    def emp_bayes(self, tail_excess):

        # sort the tail sample
        tail_data = np.sort(tail_excess)

        # set sample size and the quartile index used by the method
        sample_count = tail_data.shape[0]
        quart_pos = int(np.floor(sample_count / 4.0 + 0.5)) - 1

        # set quadrature size used to approximate the posterior mean
        grid_size = 20 + int(np.floor(np.sqrt(sample_count)))

        # theta grid
        theta_set = np.empty(grid_size)

        # store unnormalised log weights derived from the profile likelihood
        loglike_set = np.empty(grid_size)

        # fill grid and evaluate the profile likelihood at each grid point
        for grid_pos in range(grid_size):

            # set grid location using the data driven prior quantiles
            theta_set[grid_pos] = np.clip(
                1.0 / tail_data[-1] + (1.0 - np.sqrt(grid_size / (grid_pos + 0.5))) / (3.0 * tail_data[quart_pos]),
                np.finfo(float).tiny, (1.0 / tail_data[-1]) * (1.0 - np.finfo(float).eps))

            # implied shape in the Zhang Stephens parameterisation
            fit_shape = -np.mean(np.log1p(-theta_set[grid_pos] * tail_data))

            # profile log likelihood contribution for this theta
            loglike_set[grid_pos] = sample_count * (np.log(theta_set[grid_pos] / fit_shape) + fit_shape - 1.0)

        # stabilise weights using a shifted exponentiation
        max_loglike = np.max(loglike_set)
        weight_raw = np.exp(loglike_set - max_loglike)

        # normalise weights
        weight_set = weight_raw / np.sum(weight_raw)

        # posterior mean estimate of theta
        theta_fit = np.sum(theta_set * weight_set)

        # final shape and scale estimates
        fit_shape = -np.mean(np.log1p(-theta_fit * tail_data))
        fit_scale = fit_shape / theta_fit

        # convert to plus sign generalised Pareto parameterisation
        tail_shape = fit_shape
        tail_scale = fit_scale

        return tail_shape, tail_scale

    # Pareto-smoothed importance sampling leave-one-out scoring (based on Vehtari et al. 2017) - approximates the draws and predicts each removed observation
    def psis_loo(self, fixed_vals, response_vals, rand_vals, posterior_intermean, posterior_intervar,
                       posterior_fixedmeans, posterior_fixedvars, posterior_fixedcov,
                       posterior_randmeans, posterior_randvars, tau_candidates, tau_weights, fixed_mix, rand_mix,
                       draws_count=100, response_params=None):
        loo_scores = []
        loo_pit = []

        # make this cross-validated later

        # sample mixture components for joint posterior draws
        tau_index = np.random.choice(len(fixed_mix), size=draws_count, p=np.asarray(tau_weights))

        # allocate posterior draws once for reuse
        fixed_draws = np.empty((draws_count, fixed_mix[0][1].shape[0]))
        rand_draws = np.empty((draws_count, len(rand_mix[0][0])))

        # draw fixed then random effects conditional on fixed draws
        for draw_index in range(draws_count):
            mean_tau, cov_tau = fixed_mix[int(tau_index[draw_index])]
            fixed_draws[draw_index, :] = np.random.multivariate_normal(mean_tau, cov_tau)

            # get random effect summaries for this tau draw
            rand_means_tau, rand_vars_tau, cross_cov_tau = rand_mix[int(tau_index[draw_index])]

            # compute fixed shift for conditional random draws
            fixed_shift = np.linalg.solve(cov_tau, (fixed_draws[draw_index, :] - mean_tau))

            # draw random effects conditional on fixed draw
            rand_cov_solve = np.linalg.solve(cov_tau, cross_cov_tau)
            rand_mean = rand_means_tau + (cross_cov_tau.T @ fixed_shift)
            rand_var = rand_vars_tau - np.sum(cross_cov_tau * rand_cov_solve, axis=0)
            rand_draws[draw_index, :] = np.random.normal(rand_mean, np.sqrt(rand_var))

        # convert model inputs to numpy for fast row indexing
        fixed_np_all = fixed_vals.to_numpy()
        rand_np_all = rand_vals.to_numpy()
        response_np_all = response_vals.to_numpy()

        # store one Pareto-k tail-shape estimate per observation in data order
        pareto_k = np.zeros(response_np_all.shape[0])

        # draw intercepts and fixed coefficients across draws
        inter_draw = fixed_draws[:, 0]
        coeffs_draws = fixed_draws[:, 1:]

        # start from no fixed response parameter and no draw-specific response-parameter vector
        response_param = None
        response_param_draws = None

        # when response parameters were saved, determine whether they collapse to one fixed value or vary by tau
        if response_params is not None:
            response_param_values = np.asarray(response_params, dtype=float)

            # use the fast fixed-parameter path when every retained tau component has the same response parameter
            if (response_param_values.size == 1) or np.allclose(response_param_values, response_param_values[0]):
                response_param = float(response_param_values[0])

            # otherwise expand the tau-specific response parameters to the draw level
            else:
                response_param_draws = np.asarray([response_params[int(tau_index[i])] for i in range(draws_count)],
                                                 dtype=float)

        # reshape the draw-specific response parameters so the likelihood can broadcast them across held-out observations
        response_param_batch = None if response_param_draws is None else response_param_draws[:, None]

        # set the number of tail draws used for the generalised Pareto fit
        tail_count = int(np.floor(min(0.2 * draws_count, 3.0 * np.sqrt(draws_count))))

        # tail plotting positions used to approximate expected order statistics
        tail_rank = (np.arange(1, tail_count + 1) - 0.5) / tail_count

        # set the square root of the draw count for repeated use in truncation
        sqrt_draws = np.sqrt(draws_count)

        # iterate over each observation batch to amortise the fixed-effects matrix multiplication
        batch_size = 16384

        # iterate over each observation batch and calculate a log loo score
        with tqdm(total=response_np_all.shape[0]) as loo_bar:
            for batch_start in range(0, response_np_all.shape[0], batch_size):

                # extract the held-out fixed rows, random ids and observed responses for this batch
                batch_stop = min(batch_start + batch_size, response_np_all.shape[0])
                fixed_batch = fixed_np_all[batch_start:batch_stop, :]
                rand_batch = rand_np_all[batch_start:batch_stop]
                obs_batch = response_np_all[batch_start:batch_stop]

                # compute the linear predictor across draws
                linpred_batch = inter_draw[:, None] + (coeffs_draws @ fixed_batch.T) + rand_draws[:, rand_batch]

                # precompute response-only constants once per batch when the response parameter is fixed across draws
                if response_param_draws is None:
                    cache_batch = self.likelihood_model.precompute(obs_batch, response_param)
                    response_param_use = response_param

                # keep the cache disabled when the response parameter varies across draws
                else:
                    cache_batch = None
                    response_param_use = response_param_batch

                # evaluate the pointwise log probability under the likelihood across draws
                loglike_batch = self.likelihood_model.logprob(
                    obs_batch[None, :], linpred_batch, cache_batch, response_param_use)
                
                # evaluate PIT bounds under the likelihood across draws
                cdf_lower_batch, cdf_upper_batch = self.likelihood_model.pitbounds(
                    obs_batch[None, :], linpred_batch, response_param_use)

                # compute raw importance ratios as the reciprocal of the per-draw likelihood values
                raw_ratios_batch = np.exp(np.min(loglike_batch, axis=0, keepdims=True) - loglike_batch)

                # iterate over each observation and calculate a log loo score
                for local_i in range(batch_stop - batch_start):

                    # extract draw-wise values for this held-out observation
                    loglike = loglike_batch[:, local_i]
                    cdf_lower = cdf_lower_batch[:, local_i]
                    cdf_upper = cdf_upper_batch[:, local_i]
                    raw_ratios = raw_ratios_batch[:, local_i].copy()

                    # sort importance ratios to identify the upper tail used in the Pareto fit
                    tail_index = np.argpartition(raw_ratios, -tail_count)[-tail_count:]
                    tail_index = tail_index[np.argsort(raw_ratios[tail_index])]

                    # set and determine the tail threshold
                    tail_base = raw_ratios[tail_index[0]]
                    tail_excess = raw_ratios[tail_index] - tail_base
                    tail_data = np.sort(tail_excess)
                    quart_pos = int(np.floor(tail_data.shape[0] / 4.0 + 0.5)) - 1

                    # start from k = 0 when the upper tail has no variation and no Pareto smoothing is needed
                    best_shape = 0.0

                    # smooth the upper tail only when the tail excess has variation
                    if (tail_data[-1] > 0.0) and (tail_data[quart_pos] > 0.0):
                        # evaluate candidate tail values
                        best_shape, best_scale = self.emp_bayes(tail_excess)

                        # smoothed cutoffs from the fitted generalised Pareto inverse CDF
                        smoothed_excess = (best_scale / best_shape) * ((1.0 - tail_rank) ** (-best_shape) - 1.0) if best_shape != 0.0 else best_scale * (-np.log1p(-tail_rank))

                        # replace the largest raw ratios by the smoothed stats
                        raw_ratios[tail_index] = tail_base + smoothed_excess

                    # save the Pareto-k value for this held-out observation in global row order
                    pareto_k[batch_start + local_i] = best_shape

                    # truncate extreme ratios using Ionides truncated importance sampling rule
                    truncation = sqrt_draws * np.mean(raw_ratios)
                    weights = np.minimum(raw_ratios, truncation)

                    # PSIS-LOO pointwise predictive density
                    log_weights = np.full(weights.shape, -np.inf, dtype=float)
                    np.log(weights, out=log_weights, where=weights > 0.0)

                    # compute weighted predictive cdf bounds and add randomised loo pit value
                    weights_sum = np.sum(weights)
                    pit_lower = np.sum(weights * cdf_lower) / weights_sum
                    pit_upper = np.sum(weights * cdf_upper) / weights_sum
                    loo_pit.append(pit_lower + (pit_upper - pit_lower) * np.random.uniform(0.0, 1.0))

                    # PSIS-LOO log predictive density using weighted log likelihood
                    loo_scores.append(
                        scipy.special.logsumexp(loglike + log_weights) - scipy.special.logsumexp(log_weights))

                # update the progress bar once per completed batch
                loo_bar.update(batch_stop - batch_start)

        # expected log predictive density of the LOO scores
        expected_logdens = sum(loo_scores)
        avg_logdens = expected_logdens / len(loo_scores)

        # keep the pointwise Pareto-k diagnostics for later export
        self.pareto_k = pareto_k

        # plot a density overlay of the loo probability integral transform (Gabry et al., 2019)
        grid_vals = np.linspace(0.0, 1.0, 100)

        # build fixed bin edges aligned to the plotting grid
        bin_edges = np.linspace(0.0, 1.0, grid_vals.shape[0] + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = float(bin_edges[1] - bin_edges[0])

        # create a figure for the calibration plot
        loo_pit_plot_start = time.perf_counter()
        loo_fig = plt.figure()

        # keep only finite loo pit values for the density estimate
        loo_pit = np.asarray(loo_pit, dtype=float)
        loo_pit = loo_pit[np.isfinite(loo_pit)]

        # use the shared density sample size when drawing each simulated reference density
        loo_pit_plot_count = min(density_sample_size, loo_pit.shape[0])

        # keep the reference-curve sampling reproducible across runs
        plot_rng = np.random.default_rng(42)

        # draw simulated uniform reference densities for comparison
        for j in range(simulated_sets):
            sim_vals = plot_rng.uniform(0.0, 1.0, loo_pit_plot_count)
            sim_hist, _ = np.histogram(sim_vals, bins=bin_edges, density=False)

            # match the gaussian_kde bandwidth rule on this simulated sample
            sim_sd = float(np.std(sim_vals, ddof=1))
            sim_bw = sim_sd * (sim_vals.shape[0] ** (-1.0 / 5.0))
            sim_sigma = sim_bw / bin_width

            # smooth the simulated histogram on the plotting grid
            sim_smooth = scipy.ndimage.gaussian_filter1d(sim_hist.astype(float), sim_sigma, mode="constant")

            # normalise the smoothed histogram into a density curve
            sim_dens = sim_smooth / (float(np.sum(sim_smooth)) * bin_width)

            # plot the simulated uniform reference density
            plt.plot(bin_centres, sim_dens, color="lightgrey", alpha=0.4)

        # smooth the empirical loo pit density on the same grid used for the reference simulations
        pit_hist, _ = np.histogram(loo_pit, bins=bin_edges, density=False)
        pit_sd = float(np.std(loo_pit, ddof=1))
        pit_bw = pit_sd * (loo_pit.shape[0] ** (-1.0 / 5.0))
        pit_sigma = pit_bw / bin_width
        pit_smooth = scipy.ndimage.gaussian_filter1d(pit_hist.astype(float), pit_sigma, mode="constant")
        pit_dens = pit_smooth / (float(np.sum(pit_smooth)) * bin_width)

        # plot the loo pit density curve over the reference simulations
        plt.plot(bin_centres, pit_dens, "-", linewidth=2.0)

        # publication version keeps the calibration plot visually clean
        # plt.title("LOO-PIT calibration")
        # plt.xlabel("LOO-PIT")
        # plt.ylabel("Density")
        plt.xticks(np.linspace(0.0, 1.0, 6))
        plt.yticks([])
        loo_fig.savefig("loo_pit_calibration.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(loo_fig)

        # print the LOO-PIT plot runtime
        loo_pit_plot_elapsed = time.perf_counter() - loo_pit_plot_start
        loo_pit_plot_hours = int(loo_pit_plot_elapsed // 3600)
        loo_pit_plot_minutes = int((loo_pit_plot_elapsed % 3600) // 60)
        loo_pit_plot_seconds = int(loo_pit_plot_elapsed % 60)
        print(f"LOO-PIT calibration runtime: {loo_pit_plot_hours}:{loo_pit_plot_minutes}:{loo_pit_plot_seconds}")

        return avg_logdens

    """
    Results methods

    """

    # produces a credible interval for the response variable
    def credible_interval(self, fixed_vals, response_vals, rand_vals, posterior_intermean, posterior_intervar,
                          posterior_fixedmeans, posterior_fixedcov,
                          posterior_fixedvars, posterior_randmeans, posterior_randvars, tau_candidates, tau_weights,
                          fixed_mix, rand_mix, draws_count=100, batch_size=summary_batch_size, response_params=None):

        # convert inputs to arrays for numpy indexing and reuse the cached fixed-design matrix when possible
        fixed_design = self.fixed_design if fixed_vals is self.fixed_vals else np.concatenate(
            (np.ones((fixed_vals.shape[0], 1)), fixed_vals.to_numpy()), axis=1)
        rand_np = rand_vals.to_numpy()

        # draw posterior latent effects once and reuse them across observation batches
        fixed_draws, rand_draws, tau_index = self.latent_effect_draws(
            posterior_intermean, posterior_fixedmeans, posterior_fixedcov, posterior_randmeans, posterior_randvars,
            draws_count=draws_count, fixed_mix=fixed_mix, rand_mix=rand_mix, tau_weights=tau_weights,
            return_tau_index=True)

        # align the predictive response parameter draws with the sampled latent-effect draws when they are available
        response_param_draws = None
        if response_params is not None:
            response_param_draws = np.asarray([response_params[int(tau_index[i])] for i in range(draws_count)])
        
        # reshape the response parameter draws so the likelihood can broadcast them across observation batches
        response_param_batch = None if response_param_draws is None else response_param_draws[:, None]

        # convert the requested mass into the two tail probabilities
        tail_prob = (1.0 - cred_mass) / 2.0
        lower_prob = tail_prob
        upper_prob = 1.0 - tail_prob

        # allocate fitted and predictive summaries per observation
        mean = np.empty(fixed_design.shape[0])
        mean_lower = np.empty(fixed_design.shape[0])
        mean_upper = np.empty(fixed_design.shape[0])
        pred_lower = np.empty(fixed_design.shape[0])
        pred_upper = np.empty(fixed_design.shape[0])

        # compute fitted and predictive intervals in row batches to avoid storing all draws at once
        with tqdm(total=fixed_design.shape[0], desc="Credible intervals", unit="row") as interval_bar:

            # iterate over observation batches and summarise each batch before moving to the next one
            for batch_start in range(0, fixed_design.shape[0], batch_size):

                # set the end row for this batch and slice the corresponding fixed and random inputs
                batch_stop = min(batch_start + batch_size, fixed_design.shape[0])
                fixed_batch = fixed_design[batch_start:batch_stop, :]
                rand_batch = rand_np[batch_start:batch_stop]
                linpred_batch = (fixed_draws @ fixed_batch.T) + rand_draws[:, rand_batch]
                mean_draws_batch = self.likelihood_model.mean(linpred_batch)

                # store the fitted mean summaries for this batch of observations
                mean[batch_start:batch_stop] = np.mean(mean_draws_batch, axis=0)
                mean_lower[batch_start:batch_stop] = np.quantile(mean_draws_batch, lower_prob, axis=0)
                mean_upper[batch_start:batch_stop] = np.quantile(mean_draws_batch, upper_prob, axis=0)

                # draw predictive outcomes under the active likelihood instead of assuming a Poisson model
                if response_param_batch is None:
                    replicate_draws_batch = self.likelihood_model.draw_from_mean(mean_draws_batch)
                else:

                    # pass the matching response parameter draws when the likelihood needs them for prediction
                    replicate_draws_batch = self.likelihood_model.draw_from_mean(mean_draws_batch, hyper=response_param_batch)

                # store the predictive interval summaries for this batch of observations
                pred_lower[batch_start:batch_stop] = np.quantile(replicate_draws_batch, lower_prob, axis=0)
                pred_upper[batch_start:batch_stop] = np.quantile(replicate_draws_batch, upper_prob, axis=0)
                interval_bar.update(batch_stop - batch_start)

        # take the lowest and highest fitted mean bounds across observations
        mean_lower_all = np.min(mean_lower)
        mean_upper_all = np.max(mean_upper)

        # lowest and highest predictive bounds across observations
        pred_lower_all = np.min(pred_lower)
        pred_upper_all = np.max(pred_upper)

        return (mean, mean_lower, mean_upper, list(pred_lower), list(pred_upper),
                mean_lower_all, mean_upper_all, pred_lower_all, pred_upper_all)

    # summarises univariate marginals directly from INLA outputs
    def marginals(self, means, variances, lower_prob=0.025, upper_prob=0.975):

        # convert inputs into arrays for numpy indexing
        means = np.asarray(means)
        variances = np.asarray(variances)

        # compute standard deviations from the marginal variances
        std_devs = np.sqrt(variances)

        # compute equal-tailed quantiles under the marginal normal approximation
        lower_z = scipy.stats.norm.ppf(lower_prob)
        upper_z = scipy.stats.norm.ppf(upper_prob)
        lower_quants = means + (lower_z * std_devs)
        upper_quants = means + (upper_z * std_devs)

        return means, std_devs, lower_quants, upper_quants

    # composes the fixed effects results into a single dataframe object for printing and other things
    def fixed_results(self, fixed_vals, posterior_intermean, posterior_intervar, posterior_fixedmeans,
                      posterior_fixedvars):

        # rebuild the fixed-effect mean and variance vectors to include the intercept
        fixed_means = np.concatenate(([posterior_intermean], np.asarray(posterior_fixedmeans)))
        fixed_vars = np.concatenate(([posterior_intervar], np.asarray(posterior_fixedvars)))

        # summarise fixed-effect marginals directly from the INLA outputs
        fixed_means, std_devs, first_quantiles, second_quantiles = self.marginals(fixed_means, fixed_vars)

        # row names with intercept first
        row_names = ["(Intercept)"] + list(fixed_vals.columns)

        # add to dataframe
        results_df = pd.DataFrame({
            "means": fixed_means,
            "std.devs": std_devs,
            "0.025 quant": first_quantiles,
            "0.975 quant": second_quantiles
        }, index=row_names)

        return results_df

    # composes the random effects results into a single dataframe object for printing and other things
    def random_results(self, posterior_randmeans, posterior_randvars):

        # rebuild the random-effect mean and variance vectors from the INLA outputs
        rand_means = np.asarray(posterior_randmeans)
        rand_vars = np.asarray(posterior_randvars)

        # summarise random-effect marginals directly from the INLA outputs
        rand_means, std_devs, first_quantiles, second_quantiles = self.marginals(rand_means, rand_vars)

        # row names with intercept first
        row_names = list(range(1, len(rand_means) + 1))

        # add to dataframe
        results_df = pd.DataFrame({
            "means": rand_means,
            "std.devs": std_devs,
            "0.025 quant": first_quantiles,
            "0.975 quant": second_quantiles
        }, index=row_names)

        return results_df

    # composes the model precision hyperparameter results into a dataframe object
    def prec_results(self, tau_candidates, tau_weights):

        # convert inputs to numpy arrays
        tau_vals = np.asarray(tau_candidates)
        weight_vals = np.asarray(tau_weights)

        # sort tau values to idenitfy quantiles
        sort_index = np.argsort(tau_vals)
        tau_sorted = tau_vals[sort_index]
        weight_sorted = weight_vals[sort_index]

        # get weighted mean and std devs
        mean_tau = np.sum(weight_sorted * tau_sorted)
        var_tau = np.sum(weight_sorted * (tau_sorted - mean_tau) * (tau_sorted - mean_tau))
        std_tau = np.sqrt(var_tau)

        # sort tau so cum weights represent post prob mass up to each tau
        cum_weights = np.cumsum(weight_sorted)
        lower_index = np.searchsorted(cum_weights, 0.025, side="left")
        upper_index = np.searchsorted(cum_weights, 0.975, side="left")
        lower_quant = tau_sorted[min(lower_index, len(tau_sorted) - 1)]
        upper_quant = tau_sorted[min(upper_index, len(tau_sorted) - 1)]

        # add to dataframe
        row_name = ["Precision"]
        results_df = pd.DataFrame({
            "means": mean_tau,
            "std.devs": std_tau,
            "0.025 quant": lower_quant,
            "0.975 quant": upper_quant,
        },
            index=row_name)

        return results_df

    """
    Display GLMM results

    """

    # run the actual model and get the results
    def results(self):

        start = time.perf_counter()
        print("Checking INLA cache...")
        print(f"Loaded {self.rand_count} random groups and {self.obs_count} rows of data")

        # load the cached INLA solution if it exists, otherwise solve and save it immediately
        if not self.load_inla_cache():
            print("Running initial model...")
            (posterior_intermean, posterior_intervar, posterior_fixedmeans, posterior_fixedvars, posterior_fixedcov,
             posterior_randmeans, posterior_randvars,
             tau_candidates, tau_weights, fixed_mix, rand_mix, response_params) = self.inla(self.fixed_vals,
                                                                                             self.response_vals,
                                                                                             self.rand_vals)
            self.store_inla_state(posterior_intermean, posterior_intervar, posterior_fixedmeans, posterior_fixedvars,
                                  posterior_fixedcov, posterior_randmeans, posterior_randvars, tau_candidates,
                                  tau_weights, fixed_mix, rand_mix, response_params)
            self.save_inla_cache()

        posterior_intermean = self.posterior_intermean
        posterior_intervar = self.posterior_intervar
        posterior_fixedmeans = self.posterior_fixedmeans
        posterior_fixedvars = self.posterior_fixedvars
        posterior_fixedcov = self.posterior_fixedcov
        posterior_randmeans = self.posterior_randmeans
        posterior_randvars = self.posterior_randvars
        tau_candidates = self.tau_candidates
        tau_weights = self.tau_weights
        fixed_mix = self.fixed_mix
        rand_mix = self.rand_mix
        response_params = self.response_params

        # build fitted mean summaries per observation
        mean, mean_lower, mean_upper, mean_draws = self.expected_counts(self.fixed_vals, self.rand_vals,
                                                                        posterior_intermean, posterior_fixedmeans,
                                                                        posterior_fixedcov, posterior_randmeans,
                                                                        posterior_randvars)
        fitted_means = pd.DataFrame({"mean_mean": mean, "mean_lower": mean_lower, "mean_upper": mean_upper},
                                    index=self.fixed_vals.index)
        self.fitted_means = fitted_means

        # print credible and predictive interval - keep turned off - quite slow
        (mean, mean_lower, mean_upper, pred_lower, pred_upper, mean_lower_all, mean_upper_all, pred_lower_all,
         pred_upper_all) = self.credible_interval(
            self.fixed_vals, self.response_vals, self.rand_vals, posterior_intermean, posterior_intervar,
            posterior_fixedmeans, posterior_fixedcov,
            posterior_fixedvars, posterior_randmeans, posterior_randvars, tau_candidates, tau_weights, fixed_mix,
            rand_mix, response_params=response_params)
        # interpretation: the range of plausible underlying average response
        print(f"\nCredible interval (response): {cred_mass * 100:.0f}% [{mean_lower_all:.2f}, {mean_upper_all:.2f}]")
        # interpretation: the range of plausible new observed response values if the same data were observed again
        print(f"Predictive interval (response): {cred_mass * 100:.0f}% [{pred_lower_all:.2f}, {pred_upper_all:.2f}]\n")

        # get the fixed effects and print them as a table
        fixed_results = self.fixed_results(self.fixed_vals, posterior_intermean, posterior_intervar,
                                           posterior_fixedmeans, posterior_fixedvars)
        print("Fixed effects results:")
        table_fixed = fixed_results.to_string(float_format=lambda val: f"{val:.3f}")
        print(table_fixed)

        # get the random effects and print them as a table
        rand_results = self.random_results(posterior_randmeans, posterior_randvars)
        # print("\nRandom effects results:")
        # table_rand = rand_results.to_string(float_format=lambda val: f"{val:.3f}")
        # print(table_rand)

        # get the precision results and them them as a 1-row table
        precision_results = self.prec_results(tau_candidates, tau_weights)
        # print("\nModel hyperparameters:")
        # table_prec = precision_results.to_string(float_format=lambda val: f"{val:.3f}")
        # print(table_prec)

        end = time.perf_counter()
        elapsed = end - start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"\nModel time: {hours}:{minutes}:{seconds}")

        return fixed_results, rand_results, precision_results

    # posterior predictive checks
    def posterior(self):

        print("\nRunning posterior predictive checks...")
        self.check(self.fixed_vals, self.response_vals, self.rand_vals, self.posterior_intermean,
                   self.posterior_intervar, self.posterior_fixedmeans,
                   self.posterior_fixedvars, self.posterior_fixedcov, self.posterior_randmeans,
                   self.posterior_randvars, self.tau_weights, self.fixed_mix, self.rand_mix, self.response_params)
        print("\nCalculating PSIS-LOO score...")
        LOO_score = self.psis_loo(self.fixed_vals, self.response_vals, self.rand_vals, self.posterior_intermean,
                                  self.posterior_intervar, self.posterior_fixedmeans,
                                  self.posterior_fixedvars, self.posterior_fixedcov, self.posterior_randmeans,
                                  self.posterior_randvars, self.tau_candidates,
                                  self.tau_weights, self.fixed_mix, self.rand_mix, response_params=self.response_params)
        print(f"LOO score: {LOO_score}")

        return LOO_score

