"""
  Prior predictive simulation for the conditional generative model before posterior fitting.
  The script uses the real observed dataset as the observed side of the check,
  while the replicated datasets are simulated, drawn from the priors and the likelihood.

  Prior questions, anecdotally put:
  Intercept prior mean: Where do I want the model’s starting level to sit before the other terms move it around?
  Intercept prior SD: How tightly do I want to hold that starting level in place?
  Main prior mean: In which direction, on average, do I want predictors to push the model away from that starting level?
  Main prior SD: How much do I want to let predictors move it away from that starting level?
  Informativeness: the larger the SD, the less informative the prior is.
  Remember the scale: for the fixed effects, we are working on the coefficients, not the data itself

"""
import concurrent.futures
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
from tqdm import tqdm

from GLMM_rand import GLMM
from INLA_solver import inlaSolver
from gaussian_log1p import GaussianLog1pLikelihood
from gaussian_mixed import MixedGaussianLikelihood
from priors import gaussian

# Build a sparse one-hot block with the first level dropped
def build_dummy_block(values):

    # sort levels so the reference category is deterministic
    value_array = np.asarray(values, dtype=str)
    value_levels = np.sort(np.unique(value_array))

    # encode levels and keep only non-reference columns
    value_codes = pd.Categorical(value_array, categories=value_levels).codes
    row_index = np.flatnonzero(value_codes > 0)
    col_index = value_codes[row_index] - 1
    data_vals = np.ones(row_index.size, dtype=float)
    dummy_block = scipy.sparse.csr_matrix(
        (data_vals, (row_index, col_index)),
        shape=(value_array.size, value_levels.size - 1),
    )
    return dummy_block

# Build the sparse fixed-effect matrix used for prior predictive simulation
def build_fixed_effects(month, track_age, country, month_mean, month_std, age_mean, age_std, country_cols):

    # standardise month on the fitted model scale
    month = np.asarray(month, dtype=float)
    month_scaled = (month - month_mean) / month_std

    # standardise track age on the fitted model scale
    track_age = np.asarray(track_age, dtype=float)
    age_scaled = (track_age - age_mean) / age_std

    # build sparse country dummy block with the first level dropped
    dummy_country = pd.get_dummies(country, drop_first=True, prefix="Country", prefix_sep="").astype(int)
    dummy_country = dummy_country.reindex(columns=country_cols, fill_value=0).astype(int)
    country_block = scipy.sparse.csr_matrix(dummy_country.to_numpy(dtype=float))

    # build the same interaction terms as the fitted model
    age_month = age_scaled * month_scaled
    age_country = country_block.multiply(age_scaled.reshape(-1, 1))
    age_month_country = country_block.multiply(age_month.reshape(-1, 1))

    # combine terms in the same column order as GLMM_run_log1p.py
    blocks = [
        scipy.sparse.csr_matrix(month_scaled.reshape(-1, 1)),
        scipy.sparse.csr_matrix(age_scaled.reshape(-1, 1)),
        country_block,
        scipy.sparse.csr_matrix(age_month.reshape(-1, 1)),
        age_country,
        age_month_country,
    ]
    return scipy.sparse.hstack(blocks, format="csr")


# Build per-column prior vectors for the fixed effects
def fixed_prior_vectors(fixed_columns, numerical_mean, numerical_sd, dummy_mean, dummy_sd):

    # use the dummy prior for country main effects and country-involving interactions
    fixed_columns = np.asarray(fixed_columns, dtype=str)
    dummy_mask = np.char.find(fixed_columns, "Country") >= 0
    fixed_mean = np.where(dummy_mask, float(dummy_mean), float(numerical_mean))
    fixed_sd = np.where(dummy_mask, float(dummy_sd), float(numerical_sd))
    return fixed_mean, fixed_sd, dummy_mask


# Add interaction columns using dummy column groups
def interaction_table(design, groups, terms):

    # instantiate interaction terms metadata dictionary
    design.attrs["interaction_terms"] = {}
    interaction_cols = {}
    interaction_terms = design.attrs["interaction_terms"]

    # generate interaction columns by multiplying the columns implied by each interaction term
    for term in terms:

        # collect the column lists for each predictor name in the interaction
        col_lists = []
        for name in term:
            col_lists.append(groups[name])

        # create a column for every cartesian product of contributing design columns
        for col_names in itertools.product(*col_lists):

            # create an interaction column name from the column names
            term_name = "*".join(col_names)

            # prevent silent overwrites which can create collinearity
            if (term_name in design.columns) or (term_name in interaction_cols):
                raise ValueError(term_name)

            # initialise the interaction column so products can accumulate multiplicatively
            term_vals = np.ones(len(design), dtype=float)

            # multiply across all contributing columns to form the interaction term
            for col_name in col_names:
                term_vals = term_vals * design[col_name].to_numpy()

            # skip non-identifiable interaction columns with no variation
            if np.all(term_vals == term_vals[0]):
                continue

            # collect the interaction term for one join
            interaction_cols[term_name] = term_vals

            # record interaction terms metadata
            interaction_terms[term_name] = list(col_names)

    # attach the interaction terms to the design matrix
    design = pd.concat((design, pd.DataFrame(interaction_cols, index=design.index)), axis=1)
    design.attrs["interaction_terms"] = interaction_terms

    return design


# Take a random sample of tracks for fitting the mixed Gaussian likelihood
def random_sample(data, track_count, seed=42):

    # take the requested tracks from the data
    song_nums = data["Song Num"].drop_duplicates().sample(n=track_count, random_state=seed)
    data = data[data["Song Num"].isin(song_nums)]

    return data


# Build the fixed-effects table on a shared scale and column order
def fixed_table(data, month_mean, month_std, age_mean, age_std, country_cols, terms, fixed_cols, interaction_terms):

    # rebuild the numerical predictors for the supplied rows
    fixed_effects = data[["Month", "Track Age"]].copy()

    # dummy-code the countries on the supplied column structure
    dummy_country = pd.get_dummies(data["Country"], drop_first=True, prefix="Country", prefix_sep="").astype(int)
    dummy_country = dummy_country.reindex(columns=country_cols, fill_value=0).astype(int)
    fixed_effects = fixed_effects.join(dummy_country)

    # standardise the numerical predictors on the supplied scale
    fixed_effects["Month"] = (fixed_effects["Month"] - month_mean) / month_std
    fixed_effects["Track Age"] = (fixed_effects["Track Age"] - age_mean) / age_std

    # register groups for interaction expansion across the aligned columns
    groups = {}
    groups["Month"] = ["Month"]
    groups["Track Age"] = ["Track Age"]
    groups["Country"] = country_cols

    # rebuild the interaction columns on the shared scale
    fixed_effects = interaction_table(fixed_effects, groups, terms)

    # align the rebuilt design to the fitted column order
    fixed_effects = fixed_effects.reindex(columns=fixed_cols, fill_value=0.0)
    fixed_effects.attrs["interaction_terms"] = interaction_terms

    return fixed_effects


# Compute demeaned residuals on the log1p response scale
def demeaned_residuals(observed, fitted, song_num):

    # coerce observed and fitted values onto the same numeric array scale
    observed_np = np.asarray(observed, dtype=float)
    fitted_np = np.asarray(fitted, dtype=float)
    song_num_np = np.asarray(song_num)

    # residuals are observed minus fitted on the log1p response scale
    residual = observed_np - fitted_np

    # remove the mean residual within each song before measuring the observation spread
    residual_df = pd.DataFrame({"song_num": song_num_np, "residual": residual})
    residual = residual - residual_df.groupby("song_num")["residual"].transform("mean").to_numpy()

    return residual

# Load the real observed dataset and derive the model context used for checking
def load_observed_model_context(data_path, fixed_precision_prior, intercept_precision_prior,
                                pilot_variance, track_count_pilot, mixture_convergence_tolerance):

    # load the same columns used by the fitted log1p model
    data_path = Path(data_path)
    observed_data = pd.read_csv(
        data_path,
        usecols=["Country", "Luminate Song ID", "Track Age", "Month", "Streams"],
    )

    # match GLMM_run_log1p.py preprocessing
    observed_data = observed_data.rename(columns={"Luminate Song ID": "Song ID"})
    observed_data = observed_data.dropna(subset=["Country", "Song ID", "Track Age", "Month", "Streams"])
    observed_data = observed_data[observed_data["Track Age"] >= 0]

    # sort the song ids and assign ascending integers
    song_lookup = pd.DataFrame({"Song ID": np.sort(observed_data["Song ID"].unique())})
    song_lookup["Song Num"] = np.arange(1, len(song_lookup) + 1)
    observed_data = observed_data.merge(song_lookup, on="Song ID", how="left")

    # define fixed effects on the same scale and with the same interactions as GLMM_run_log1p.py
    fixed_effects = observed_data[["Month", "Track Age"]].copy()
    dummy_country = pd.get_dummies(observed_data["Country"], drop_first=True, prefix="Country", prefix_sep="").astype(int)
    fixed_effects = fixed_effects.join(dummy_country)
    month_mean = fixed_effects["Month"].mean()
    month_std = fixed_effects["Month"].std()
    age_mean = fixed_effects["Track Age"].mean()
    age_std = fixed_effects["Track Age"].std()
    fixed_effects["Month"] = (fixed_effects["Month"] - month_mean) / month_std
    fixed_effects["Track Age"] = (fixed_effects["Track Age"] - age_mean) / age_std
    country_cols = list(dummy_country.columns)
    groups = {}
    groups["Month"] = ["Month"]
    groups["Track Age"] = ["Track Age"]
    groups["Country"] = country_cols
    terms = [("Track Age", "Month"),
             ("Track Age", "Country"),
             ("Track Age", "Month", "Country")]
    fixed_effects = interaction_table(fixed_effects, groups, terms)
    fixed_cols = fixed_effects.columns
    interaction_terms = fixed_effects.attrs["interaction_terms"]

    # build sparse fixed-effect inputs for prior predictive draws
    fixed_values = build_fixed_effects(
        month=observed_data["Month"].to_numpy(dtype=float),
        track_age=observed_data["Track Age"].to_numpy(dtype=float),
        country=observed_data["Country"].to_numpy(dtype=str),
        month_mean=month_mean,
        month_std=month_std,
        age_mean=age_mean,
        age_std=age_std,
        country_cols=country_cols,
    )

    # fit the same pilot mixed-Gaussian likelihood used by the final log1p model
    prior = gaussian(fixed_precision_prior)
    pilot_data = random_sample(observed_data, track_count_pilot)
    pilot_fixed_effects = fixed_table(pilot_data, month_mean, month_std, age_mean, age_std, country_cols, terms,
                                      fixed_cols, interaction_terms)
    pilot_linear_model = GLMM()
    pilot_group_index, pilot_rand_count = pilot_linear_model.prepare_groups(pilot_data["Song Num"])
    pilot_random_effects = pd.Series(pilot_group_index, index=pilot_data.index)
    pilot_response = np.log1p(pilot_data["Streams"])
    pilot_likelihood_model = GaussianLog1pLikelihood(variance=pilot_variance)
    pilot_inla = inlaSolver(pilot_linear_model, pilot_likelihood_model, prior, intercept_precision_prior,
                            fixed_precision_prior, pilot_fixed_effects, pilot_response, pilot_random_effects)
    (pilot_intermean, pilot_intervar, pilot_fixedmeans, pilot_fixedvars, pilot_fixedcov, pilot_randmeans,
     pilot_randvars, pilot_tau_candidates, pilot_tau_weights, pilot_fixed_mix, pilot_rand_mix,
     pilot_response_params) = pilot_inla.inla(pilot_fixed_effects, pilot_response, pilot_random_effects)
    pilot_fixed_np = pilot_fixed_effects.to_numpy(dtype=float)
    pilot_rand_np = pilot_random_effects.to_numpy()
    pilot_fitted = (float(pilot_intermean) + (pilot_fixed_np @ np.asarray(pilot_fixedmeans, dtype=float)) +
                    np.asarray(pilot_randmeans, dtype=float)[pilot_rand_np])
    pilot_residual = demeaned_residuals(pilot_response, pilot_fitted, pilot_data["Song Num"])
    likelihood_model = MixedGaussianLikelihood(residual_vals=pilot_residual,
                                               convergence_tolerance=mixture_convergence_tolerance)

    # build random-effect group ids for prior predictive simulation
    linear_model = GLMM()
    group_index, rand_count = linear_model.prepare_groups(observed_data["Song Num"])

    # keep the observed outcome columns used in the check
    observed_outcomes = observed_data[["Country", "Track Age", "Streams", "Song Num"]].copy()
    print(f"Loaded observed rows: {len(observed_outcomes)}")
    print(f"Fixed-effect columns: {fixed_values.shape[1]}")
    print(f"Random-effect groups: {rand_count}")
    print(f"Narrow component weight: {likelihood_model.mixture_weight:.4f}")
    print(f"Narrow component standard deviation: {np.sqrt(likelihood_model.narrow_variance):.4f}")
    print(f"Wide component standard deviation: {np.sqrt(likelihood_model.wide_variance):.4f}")

    return observed_outcomes, fixed_values, fixed_cols, group_index, rand_count, likelihood_model

# Take a manageable sample for density overlays when the observed dataset is large
def sample_for_overlay(values, sample_size, rng):

    # convert the values and return them unchanged when sampling is not required
    value_array = np.asarray(values)
    if (sample_size is None) or (value_array.size <= sample_size):
        return value_array

    # draw a sample without replacement and return the selected values
    sample_index = rng.choice(value_array.size, size=sample_size, replace=False)
    return value_array[sample_index]

# Run one prior predictive draw using shared design inputs
def run_prior_predictive_draw(task):

    # split the task tuple into draw inputs and start the random generator for this draw
    (draw_seed, fixed_matrix, group_index, rand_count, response_values, fixed_mean, fixed_sd,
     intercept_prior_mean, intercept_prior_sd, gamma_shape, gamma_rate, collect_overlay, overlay_sample_size,
     likelihood_model) = task
    rng = np.random.default_rng(draw_seed)

    # Draw all unknown quantities from priors
    intercept = rng.normal(loc=intercept_prior_mean, scale=intercept_prior_sd)
    coefficients = rng.normal(loc=fixed_mean, scale=fixed_sd)

    # draw random intercepts from the same gamma-normal hierarchy used by GLMM_rand.py
    tau = rng.gamma(shape=gamma_shape, scale=1.0 / gamma_rate)
    rand_intercepts = rng.normal(loc=0.0, scale=1.0 / np.sqrt(tau), size=rand_count)

    # Build the linear predictor in place to reduce peak allocation cost
    linear_predictor = fixed_matrix @ coefficients
    linear_predictor += intercept
    linear_predictor += rand_intercepts[group_index]

    # Draw from the likelihood and reuse the fitted mean in both tail summaries
    mean_values = likelihood_model.mean(linear_predictor)
    replicate = likelihood_model.draw_from_mean(mean_values, rng=rng)
    replicate_plot = likelihood_model.ppc_transform(replicate)

    # take the subsample used for the density overlay plot
    overlay_values = None
    if collect_overlay:
        overlay_values = sample_for_overlay(replicate, overlay_sample_size, rng)

    # collect the summary statistics and likelihood totals for this draw
    result = {
        "rep_mean": float(np.mean(replicate_plot)),
        "rep_sd": float(np.std(replicate_plot, ddof=1)),
        "overlay_values": overlay_values,
        "obs_loglik": likelihood_model.logsum(response_values, linear_predictor),
        "rep_loglik": likelihood_model.logsum(replicate, linear_predictor),
    }
    return result

# Draw parameter sets from priors and simulate replicated outcomes
def prior_predictive_check(observed_data, fixed_values, fixed_columns, group_index, rand_count, likelihood_model,
                           draws_count, overlay_count, overlay_sample_size, numerical_mean, numerical_sd,
                           dummy_mean, dummy_sd, intercept_prior_mean, intercept_prior_precision, gamma_shape,
                           gamma_rate, seed):

    # use the observed stream counts on the same log1p scale as the fitted model
    response_values = np.log1p(np.asarray(observed_data["Streams"], dtype=float))
    fixed_matrix = fixed_values.tocsr()

    # derive the fixed-effect prior vectors used by the prior predictive draws
    fixed_mean, fixed_sd, dummy_mask = fixed_prior_vectors(
        fixed_columns, numerical_mean, numerical_sd, dummy_mean, dummy_sd
    )
    intercept_prior_precision = float(intercept_prior_precision)
    intercept_prior_mean = likelihood_model.default_intercept(response_values) if intercept_prior_precision == 0.0 else float(intercept_prior_mean) # turn this line off to use mean with 0 precision
    intercept_prior_sd = 0.0 if intercept_prior_precision == 0.0 else 1.0 / np.sqrt(intercept_prior_precision)

    # Store replicated summary statistics for observed comparison
    rep_means = np.empty(draws_count)
    rep_sds = np.empty(draws_count)
    overlay_sets = []

    obs_loglik = np.empty(draws_count)
    rep_loglik = np.empty(draws_count)

    # create one seed and one task tuple for each prior predictive draw
    draw_seeds = np.random.SeedSequence(seed).spawn(draws_count)
    draw_tasks = ((draw_seeds[index], fixed_matrix, group_index, rand_count, response_values,
                   fixed_mean, fixed_sd, intercept_prior_mean, intercept_prior_sd, gamma_shape, gamma_rate,
                   index < overlay_count, overlay_sample_size, likelihood_model) for index in range(draws_count))

    # set up the draw execution path
    executor = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=14)
    draw_results = executor.map(run_prior_predictive_draw, draw_tasks)

    # fill the stored summaries from each completed draw
    try:
        result_iterator = tqdm(draw_results, total=draws_count, desc="Prior predictive draws", unit="draw")
        for index, draw_result in enumerate(result_iterator):
            rep_means[index] = draw_result["rep_mean"]
            rep_sds[index] = draw_result["rep_sd"]
            obs_loglik[index] = draw_result["obs_loglik"]
            rep_loglik[index] = draw_result["rep_loglik"]

            # store the overlay sample when this draw collected one
            if draw_result["overlay_values"] is not None:
                overlay_sets.append(draw_result["overlay_values"])

    # shut down the executor
    finally:
        executor.shutdown()

    # Compare observed mean and observed standard deviation to replicated distributions
    obs_vals = likelihood_model.ppc_transform(np.asarray(response_values, dtype=float))
    obs_mean = float(np.mean(obs_vals))
    obs_sd = float(np.std(obs_vals, ddof=1))
    summary = {
        "observed_mean": obs_mean,
        "observed_sd": obs_sd,
        "replicated_mean_2.5%": float(np.percentile(rep_means, 2.5)),
        "replicated_mean_50%": float(np.percentile(rep_means, 50.0)),
        "replicated_mean_97.5%": float(np.percentile(rep_means, 97.5)),
        "replicated_sd_2.5%": float(np.percentile(rep_sds, 2.5)),
        "replicated_sd_50%": float(np.percentile(rep_sds, 50.0)),
        "replicated_sd_97.5%": float(np.percentile(rep_sds, 97.5)),
        "prior_predictive_p_mean": float(np.mean(rep_means >= obs_mean)),
        "prior_predictive_p_sd": float(np.mean(rep_sds >= obs_sd)),
        "obs_loglik_2.5%": float(np.percentile(obs_loglik, 2.5)),
        "obs_loglik_50%": float(np.percentile(obs_loglik, 50.0)),
        "obs_loglik_97.5%": float(np.percentile(obs_loglik, 97.5)),
        "rep_loglik_2.5%": float(np.percentile(rep_loglik, 2.5)),
        "rep_loglik_50%": float(np.percentile(rep_loglik, 50.0)),
        "rep_loglik_97.5%": float(np.percentile(rep_loglik, 97.5)),
        "numerical_mean": float(numerical_mean),
        "numerical_sd": float(numerical_sd),
        "numerical_precision": float(1.0 / (numerical_sd ** 2)),
        "dummy_mean": float(dummy_mean),
        "dummy_sd": float(dummy_sd),
        "dummy_precision": float(1.0 / (dummy_sd ** 2)),
        "dummy_prior_columns": int(np.sum(dummy_mask)),
        "numerical_prior_columns": int(np.sum(~dummy_mask)),
        "intercept_prior_mean": intercept_prior_mean,
        "intercept_prior_precision": intercept_prior_precision,
        "random_precision_shape": gamma_shape,
        "random_precision_rate": gamma_rate,
        "mixture_weight": likelihood_model.mixture_weight,
        "narrow_variance": likelihood_model.narrow_variance,
        "wide_variance": likelihood_model.wide_variance,
    }
    return response_values, overlay_sets, summary

# Evaluate one density curve on the shared grid
def evaluate_density_curve(task):

    # split the task tuple into the transformed values and the evaluation grid
    values, grid = task

    # fit the kernel density and evaluate it on the shared grid
    density = scipy.stats.gaussian_kde(values)(grid)
    return density

# Overlay observed density with individual replicated curves and a replicated median curve
def plot_overlay(response_values, overlay_sets, density_sample_size, seed, likelihood_model, summary, output_path):

    # sample the observed and simulated values on the modelled log1p scale
    rng = np.random.default_rng(seed)
    observed_plot = sample_for_overlay(response_values, density_sample_size, rng)
    simulated_plot = [
        sample_for_overlay(values, density_sample_size, rng)
        for values in overlay_sets
    ]

    # build a robust plotting grid so one extreme draw does not stretch the full axis
    all_values = [observed_plot] + simulated_plot
    pooled_values = np.concatenate(all_values)
    lower = np.percentile(pooled_values, 0.5)
    upper = np.percentile(pooled_values, 99.5)
    grid = np.linspace(lower, upper, 500)

    # create the figure and evaluate all replicated density curves in parallel
    figure, axis = plt.subplots()
    executor = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=14)
    density_tasks = ((values, grid) for values in simulated_plot)
    density_results = executor.map(evaluate_density_curve, density_tasks)

    # collect the replicated densities so the plot can show individual runs and their median
    try:
        density_iterator = tqdm(density_results, total=len(simulated_plot), desc="Overlay densities", unit="curve")
        density_list = [density for density in density_iterator]

    # shut down the executor
    finally:
        executor.shutdown()

    # summarise the replicated densities with a replicated median curve
    density_array = np.vstack(density_list)
    density_mid = np.percentile(density_array, 50.0, axis=0)

    # draw the individual replicated curves and the replicated median curve
    for density_curve in density_list:
        axis.plot(grid, density_curve, color="lightgrey", alpha=0.4)
    axis.plot(grid, density_mid, color="grey", linewidth=1.5)

    # draw the observed density curve
    observed_density = scipy.stats.gaussian_kde(observed_plot)(grid)
    axis.plot(grid, observed_density, color="black", linewidth=2.0)

    # axis.set_title(
    #     "Prior Predictive Check\n"
    #     f"Numerical mean={summary['numerical_mean']:.4g}, precision={summary['numerical_precision']:.4g}; "
    #     f"Dummy mean={summary['dummy_mean']:.4g}, precision={summary['dummy_precision']:.4g}\n"
    #     f"Intercept mean={summary['intercept_prior_mean']:.4g}, precision={summary['intercept_prior_precision']:.4g}"
    # )
    # axis.set_xlabel("log1p(Streams)")
    # axis.set_ylabel("Density on log1p scale")
    # axis.legend()
    axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)

# Save the prior predictive summary metrics to csv
def save_summary_csv(summary, output_path):

    # write the summary as a one-row table for downstream use
    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(output_path, index=False)

# convert the prior moments to the log scale
def log_match(mean, sd):

    # convert the standard deviation to variance
    var = sd ** 2

    # match the moments on the log scale
    log_var = np.log(1.0 + (var / (mean ** 2)))
    log_mean = np.log(mean) - (0.5 * log_var)

    # convert the log variance to precision
    log_prec = 1.0 / log_var
    return log_mean, log_prec

# fixed-effect prior on the log1p linear predictor scale
numerical_mean = 0.0
numerical_sd = 0.5
dummy_mean = 0.0
dummy_sd = 0.5

# intercept prior used by GLMM_run_log1p.py
intercept_prior_mean = 0.0
intercept_precision_prior = 0.0

# random-effect prior defaults inherited by GLMM_run_log1p.py through GLMM()
default_linear_model = GLMM()
gamma_shape = default_linear_model.gamma_shape
gamma_rate = default_linear_model.gamma_rate

# mixed-Gaussian likelihood settings used by GLMM_run_log1p.py
pilot_variance = 0.2
mixture_convergence_tolerance = 1.0e-8
track_count_pilot = 5000

# load the observed dataset and set file names
# date_version = "_30_days"
date_version = "_cyear"
data_path = Path(__file__).resolve().parents[1] / "Data" / f"main_data_complete{date_version}.csv"
plot_path = Path(__file__).resolve().parent / f"prior_predictive_check{date_version}.png"
summary_csv_path = Path(__file__).resolve().parent / f"prior_predictive_summary{date_version}.csv"
numerical_precision = 1.0 / (numerical_sd ** 2)

# load observed data and fit the pilot likelihood used by the final model
observed_data, fixed_values, fixed_columns, group_index, rand_count, likelihood_model = load_observed_model_context(
    data_path,
    fixed_precision_prior=numerical_precision,
    intercept_precision_prior=intercept_precision_prior,
    pilot_variance=pilot_variance,
    track_count_pilot=track_count_pilot,
    mixture_convergence_tolerance=mixture_convergence_tolerance,
)

# run the prior predictive check against the observed outcomes
observed_values, overlay_sets, summary = prior_predictive_check(
    observed_data, fixed_values, fixed_columns, group_index, rand_count, likelihood_model,
    draws_count=2000,
    overlay_count=100,
    overlay_sample_size=20000,
    numerical_mean=numerical_mean,
    numerical_sd=numerical_sd,
    dummy_mean=dummy_mean,
    dummy_sd=dummy_sd,
    intercept_prior_mean=intercept_prior_mean,
    intercept_prior_precision=intercept_precision_prior,
    gamma_shape=gamma_shape,
    gamma_rate=gamma_rate,
    seed=42
)

# plot the checks
plot_overlay(
    observed_values,
    overlay_sets,
    density_sample_size=50000,
    seed=42,
    likelihood_model=likelihood_model,
    summary=summary,
    output_path=plot_path,
)
save_summary_csv(summary, summary_csv_path)
print("Prior predictive comparison summary")
for name, value in summary.items():
    print(f"{name}: {value}")
print(f"Saved prior predictive plot to: {plot_path}")
print(f"Saved prior predictive summary to: {summary_csv_path}")
