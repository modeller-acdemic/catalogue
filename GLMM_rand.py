# -*- coding: utf-8 -*-
"""
Random intercept GLMM for the INLA solver

@author: Chris
"""

import numpy as np
import pandas as pd
import math


class GLMM:

    # Set gamma hyperparameters and the starting value for tau
    def __init__(self, gamma_shape=1.0, gamma_rate=0.00005, tau_start=math.exp(4.0)):
        self.gamma_shape = float(gamma_shape)
        self.gamma_rate = float(gamma_rate)
        self.tau_start = float(tau_start)

        self.group_cache = None
        return

    # Recode group ids to consecutive integers and cache indices used for within-group sums
    def prepare_groups(self, group_ids):

        # Coerce inputs into a pandas Series to preserve order
        group_series = pd.Series(group_ids)

        # Map ids to consecutive integer codes in first-appearance order
        group_codes, _ = pd.factorize(group_series, sort=False)

        # Coerce to a numpy integer array for indexing
        group_index = np.asarray(group_codes, dtype=int)

        # Compute the number of groups
        group_count = int(np.max(group_index)) + 1

        # Cache indices used for within-group sums
        self.cache_reuse(group_index, group_count)

        return group_index, group_count

    # Return the random intercept term for each observation
    def random_offset(self, rand_inters, group_index):
        return np.asarray(rand_inters, dtype=float)[np.asarray(group_index, dtype=int)]

    # Return the starting value for tau
    def tau_initial(self):
        return float(self.tau_start)

    # Return the log density of the gamma prior for tau
    def tau_logprior(self, tau):
        return float((self.gamma_shape - 1.0) * np.log(tau) - (self.gamma_rate * tau))

    # Return the log density of the normal prior for the random intercepts given tau
    def rand_logprior(self, rand_inters, tau):
        rand_inters = np.asarray(rand_inters, dtype=float)
        return float((0.5 * rand_inters.size * np.log(tau)) - (0.5 * tau * float(np.dot(rand_inters, rand_inters))))

    # Compute sums within each group of likelihood weights and likelihood score values
    def group_summaries(self, weights_obs, score_obs, group_index, group_count):

        # Coerce inputs into numpy arrays for within-group sums
        weights_obs = np.asarray(weights_obs, dtype=float)
        score_obs = np.asarray(score_obs, dtype=float)
        group_index = np.asarray(group_index, dtype=int)

        # Get cached indices used for within-group sums
        perm, starts = self.cache_reuse(group_index, int(group_count))

        # Sort observations by group id
        weights_sort = weights_obs[perm]
        score_sort = score_obs[perm]

        # Sum values within each group
        weights_sum = np.add.reduceat(weights_sort, starts)
        score_sum = np.add.reduceat(score_sort, starts)

        return weights_sum, score_sum

    # Return the diagonal entries of the conditional precision matrix for the random intercepts
    def random_precision_diag(self, weights_sum, tau):
        return np.asarray(weights_sum, dtype=float) + float(tau)

    # Build the matrix of sums within each group for the intercept and each fixed column, weighted by the likelihood weights
    def cross_precision_terms(self, fixed_design, weights_obs, group_index, group_count):

        # Coerce inputs into numpy arrays for within-group sums
        fixed_design = np.asarray(fixed_design, dtype=float)
        weights_obs = np.asarray(weights_obs, dtype=float)
        group_index = np.asarray(group_index, dtype=int)

        # Get cached indices used for within-group sums
        perm, starts = self.cache_reuse(group_index, int(group_count))

        # Allocate the matrix with a row for the intercept and each fixed column
        fixed_count = int(fixed_design.shape[1])
        cross_terms = np.empty((fixed_count, int(group_count)), dtype=float)

        # Compute within-group sums of weights for the intercept row
        weights_sort = weights_obs[perm]
        weights_sum = np.add.reduceat(weights_sort, starts)

        # Store the intercept row
        cross_terms[0, :] = weights_sum

        # Compute within-group sums for each fixed column after weighting by the likelihood weights
        for col_index in range(1, fixed_count):

            # Build the weighted fixed column in observation order
            weighted_col = weights_obs * fixed_design[:, col_index]

            # Sort observations by group id
            weighted_sort = weighted_col[perm]

            # Sum values within each group for this fixed column
            cross_terms[col_index, :] = np.add.reduceat(weighted_sort, starts)

        return cross_terms

    # Return the score for the random intercepts under the conditional posterior
    def random_score(self, score_sum, rand_inters, tau):
        return np.asarray(score_sum, dtype=float) - (float(tau) * np.asarray(rand_inters, dtype=float))

    # Return the number of groups
    def random_structure_size(self, group_count):
        return int(group_count)

    # Cache or reuse the indices used to sort observations by group id and identify group boundaries
    def cache_reuse(self, group_index, group_count):

        # Define a cache key from the observation count and number of groups
        cache_key = (int(group_index.shape[0]), int(group_count))

        # Read the current cache state once
        cache = self.group_cache

        # Build and store indices when the cached value does not match this grouping layout
        if (cache is None) or (cache["key"] != cache_key):

            # Compute indices that sort observations by group id
            perm = np.argsort(group_index)

            # Build the group id vector in sorted order
            group_sort = group_index[perm]

            # Identify the first index of each group in sorted order
            starts = np.concatenate(([0], np.flatnonzero(np.diff(group_sort)) + 1))

            # Store cached indices
            self.group_cache = {"key": cache_key, "perm": perm, "starts": starts}

        # Return cached indices
        perm = self.group_cache["perm"]
        starts = self.group_cache["starts"]

        return perm, starts
