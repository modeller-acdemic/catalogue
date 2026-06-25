# -*- coding: utf-8 -*-
"""
Runs the same mixed Gaussian model as GLMM_run, but fits it separately for every country and then collates the fitted slopes into one table and plot.
This version uses the full final-run dataset to keep the Month and Track Age scaling fixed across countries, so the country lines can be compared on the same basis.

@author: Chris
"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import itertools
from GLMM_rand import GLMM
from INLA_solver import inlaSolver
from gaussian_log1p import GaussianLog1pLikelihood
from gaussian_mixed import MixedGaussianLikelihood
from tqdm import tqdm

# get muted line colours
def muted_colours(count):

    # define a restrained palette with visible separation between lines
    palette = np.array([
        "#2f6f9f", "#4f8f4f", "#a24f4f", "#7a5fa3", "#24918f",
        "#a87922", "#5d6f9f", "#9a6a4f", "#6f8f3f", "#b05878",
        "#3a7f8f", "#6a6a6a",
    ])

    # repeat the palette if the country count exceeds the colour count
    colour_vals = np.resize(palette, count)
    return colour_vals

# label the line ends without overlapping country names
def label_ends(axis, label_data, line_x, label_x, font_size):

    # sort the labels from bottom to top
    label_data = sorted(label_data, key=lambda item: item["y"])
    raw_y_vals = np.array([label_row["y"] for label_row in label_data], dtype=float)

    # get the full y-range needed for the plotted lines and the unadjusted labels
    y_min = min(float(raw_y_vals.min()), float(axis.dataLim.ymin))
    y_max = max(float(raw_y_vals.max()), float(axis.dataLim.ymax))
    y_range = max(y_max - y_min, 0.001)

    # apply an initial y-axis range so matplotlib can report the axis height
    axis.set_ylim(y_min - (0.08 * y_range), y_max + (0.08 * y_range))
    axis.figure.canvas.draw()

    # convert the font height into y-axis units
    axis_height_pixels = axis.get_window_extent().height
    point_height_pixels = font_size * axis.figure.dpi / 72.0
    label_gap = (point_height_pixels * 1.18) * y_range / max(axis_height_pixels, 1.0)

    label_y = []

    # move labels upward when adjacent labels would overlap
    for label_row in label_data:
        y_pos = float(label_row["y"])

        # keep the current label at least one label height above the previous label
        if label_y and y_pos < (label_y[-1] + label_gap):
            y_pos = label_y[-1] + label_gap
        label_y.append(y_pos)

    # include the adjusted labels in the final y-axis range
    y_min = min(float(axis.dataLim.ymin), float(raw_y_vals.min()), float(min(label_y)))
    y_max = max(float(axis.dataLim.ymax), float(raw_y_vals.max()), float(max(label_y)))

    # add the larger of ordinary plot padding and label padding
    y_range = max(y_max - y_min, 0.001)
    y_pad = max(0.08 * y_range, 0.6 * label_gap, 0.001)

    # apply the expanded y-axis limits
    axis.set_ylim(y_min - y_pad, y_max + y_pad)

    # add endpoint circles on the final year line
    for label_row in label_data:
        axis.scatter(line_x, float(label_row["y"]), color=label_row["colour"], s=27, zorder=5, clip_on=False)

    # draw the adjusted country labels
    for label_row, y_pos in zip(label_data, label_y):
        axis.text(label_x, y_pos, label_row["country"], color=label_row["colour"], fontsize=font_size,
                  fontweight="bold", va="center", ha="left", clip_on=False,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))

    return

# plot the collated country slopes and save the plotted values to csv
def plot_results(slope_data, output_dir):

    font_size = 14
    slope_path = os.path.join(output_dir, "country_slopes_pooled.csv")

    # update the existing full plot table when only selected countries were rerun
    if os.path.isfile(slope_path):
        previous_slopes_wide = pd.read_csv(slope_path)
        previous_year_cols = [col for col in previous_slopes_wide.columns if str(col).isdigit()]
        previous_slopes = previous_slopes_wide.melt(id_vars="Country", value_vars=previous_year_cols,
                                                    var_name="Year", value_name="age_slope")
        previous_slopes["Year"] = previous_slopes["Year"].astype(int)

        for suffix, value_name in (("_lower", "age_slope_lower"), ("_upper", "age_slope_upper")):
            interval_cols = [f"{year_col}{suffix}" for year_col in previous_year_cols
                             if f"{year_col}{suffix}" in previous_slopes_wide.columns]
            if interval_cols:
                previous_interval = previous_slopes_wide.melt(id_vars="Country", value_vars=interval_cols,
                                                              var_name="Year", value_name=value_name)
                previous_interval["Year"] = previous_interval["Year"].str.replace(suffix, "", regex=False).astype(int)
                previous_slopes = previous_slopes.merge(previous_interval, on=["Country", "Year"], how="left")

        updated_countries = slope_data["Country"].unique()
        previous_slopes = previous_slopes[~previous_slopes["Country"].isin(updated_countries)]
        slope_data = pd.concat((previous_slopes, slope_data), ignore_index=True)

    # get level lists from the collated slope rows
    country_list = sorted(slope_data["Country"].unique())
    year_list = sorted(slope_data["Year"].unique())
    time_labels = np.arange(1, len(year_list) + 1)

    # reshape the plotted slopes into a country-by-year table
    slope_wide = slope_data.pivot(index="Country", columns="Year", values="age_slope").reindex(country_list)
    slope_export = pd.DataFrame({"Country": slope_wide.index})
    has_intervals = {"age_slope_lower", "age_slope_upper"}.issubset(slope_data.columns)

    if has_intervals:
        lower_wide = slope_data.pivot(index="Country", columns="Year", values="age_slope_lower").reindex(country_list)
        upper_wide = slope_data.pivot(index="Country", columns="Year", values="age_slope_upper").reindex(country_list)

    for year in year_list:
        slope_export[str(year)] = slope_wide[year].to_numpy(dtype=float)
        if has_intervals:
            slope_export[f"{year}_lower"] = lower_wide[year].to_numpy(dtype=float)
            slope_export[f"{year}_upper"] = upper_wide[year].to_numpy(dtype=float)

    # save the plotted slopes in the same table layout used for the figure
    slope_export.round(3).to_csv(slope_path, index=False)

    # create figure and axes
    figure, axis = plt.subplots(figsize=(12, 8))
    colour_vals = muted_colours(len(country_list))

    # draw one line per country and store its final point
    line_ends = []
    for index, country in enumerate(country_list):
        country_plot_data = slope_data[slope_data["Country"] == country].sort_values("Year")
        slope_vals = country_plot_data["age_slope"].to_numpy(dtype=float)
        axis.plot(time_labels, slope_vals, color=colour_vals[index], linewidth=1.8)
        line_ends.append({"country": country, "y": slope_vals[-1], "colour": colour_vals[index]})

    # draw the country labels to the right of the line ends
    label_x = time_labels[-1] + 0.15
    label_ends(axis, line_ends, time_labels[-1], label_x, font_size=font_size)

    # force x ticks onto evenly-spaced year positions
    axis.set_xticks(time_labels)
    axis.set_xticklabels(year_list)
    axis.set_xlim(time_labels[0], time_labels[-1])
    axis.tick_params(axis="both", labelsize=font_size)

    # housekeeping
    plt.tight_layout()
    figure.savefig(os.path.join(output_dir, "plot_results_pooled.png"), dpi=300, bbox_inches="tight")
    plt.show()

    return

# add interaction columns using dummy column groups
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
            interaction_terms[term_name] = list(col_names)

    # attach the interaction terms to the design matrix
    design = pd.concat((design, pd.DataFrame(interaction_cols, index=design.index)), axis=1)
    design.attrs["interaction_terms"] = interaction_terms

    return design

# take a random sample of the data for testing
def random_sample(data, track_count, seed=42):

    # take the requested tracks from the data
    song_nums = data["Song Num"].drop_duplicates().sample(n=track_count, random_state=seed)
    data = data[data["Song Num"].isin(song_nums)]

    return data

# build the fixed-effects table on a shared scale and column order
def fixed_table(data, month_mean, month_std, age_mean, age_std, terms, fixed_cols, interaction_terms):

    # rebuild the numerical predictors for the supplied rows
    fixed_effects = data[["Month", "Track Age"]].copy()

    # standardise the numerical predictors on the supplied scale
    fixed_effects["Month"] = (fixed_effects["Month"] - month_mean) / month_std
    fixed_effects["Track Age"] = (fixed_effects["Track Age"] - age_mean) / age_std

    # register groups for interaction expansion across the aligned columns
    groups = {}
    groups["Month"] = ["Month"]
    groups["Track Age"] = ["Track Age"]

    # rebuild the interaction columns on the shared scale
    fixed_effects = interaction_table(fixed_effects, groups, terms)

    # align the rebuilt design to the fitted column order
    fixed_effects = fixed_effects.reindex(columns=fixed_cols, fill_value=0.0)
    fixed_effects.attrs["interaction_terms"] = interaction_terms

    return fixed_effects

# build the fixed-effect prior precision vector
def prior_vector(fixed_cols, numerical_sd, dummy_sd):

    # convert the fixed-effect column names into a pandas index
    fixed_cols = pd.Index(fixed_cols)

    # mark country main effects and country interaction terms
    dummy_mask = fixed_cols.str.contains("Country", regex=False)

    # start every fixed-effect column on the numerical prior precision
    precision_vals = np.repeat(1.0 / (numerical_sd ** 2), len(fixed_cols)).astype(float)

    # replace country-related columns with the dummy prior precision
    precision_vals[dummy_mask] = 1.0 / (dummy_sd ** 2)

    # return the precision vector in the fixed-effect column order
    return precision_vals

# gaussian prior with a fixed-effect precision vector
class gaussian_prior:

    # store the fixed-effect prior precision
    def __init__(self, fixed_precision_prior):

        # convert the supplied precision to a numpy array
        self.fixed_precision_prior = np.asarray(fixed_precision_prior, dtype=float)
        return

    # evaluate the zero-centred normal log prior
    def log_prior(self, intercept, values, fixed_vals=None, precision=None):

        # use the stored fixed precision when none was supplied
        precision = self.fixed_precision_prior if precision is None else precision

        # convert values and precision into numpy arrays
        values_np = np.asarray(intercept, dtype=float)
        precision_np = np.asarray(precision, dtype=float)

        # broadcast scalar precision across vector inputs
        if precision_np.shape == ():
            precision_np = np.full(values_np.shape, float(precision_np))

        # keep only coefficients with proper normal priors
        proper_mask = precision_np > 0.0

        # return a constant when every supplied precision is flat
        if not np.any(proper_mask):
            return 0.0

        # compute the quadratic term under the zero-centred normal prior
        quad = np.sum(precision_np[proper_mask] * np.square(values_np[proper_mask]))

        # compute the normalising term for the retained coefficients
        norm = 0.5 * np.sum(np.log(precision_np[proper_mask])) - (0.5 * np.sum(proper_mask) * np.log(2.0 * np.pi))

        # return the summed log prior density
        prior = norm - (0.5 * quad)
        return float(prior)

# compute demeaned residuals on the log1p response scale
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

"""
Data and model setup

Model parameters:
response: airplay
fixed effects: day_index and age_days
random effect (groupings): song_id - MUST be an ascending integer for the above model to work
interaction term: age*time

"""
start = time.perf_counter()

print("Loading data...")

# load in the full model data
full_data = pd.read_csv("../Data/main_data_complete_cyear.csv",
                        usecols=["Country", "Luminate Song ID", "Track Age", "Month", "Streams"])

# rename the model columns
full_data = full_data.rename(columns={"Luminate Song ID": "Song ID"})

# remove rows with missing model data
full_data = full_data.dropna(subset=["Country", "Song ID", "Track Age", "Month", "Streams"])

# remove rows with negative track age
full_data = full_data[full_data["Track Age"] >= 0]

# keep the full-data scaling used by the final run
full_fixed = full_data[["Month", "Track Age"]].copy()
month_mean = full_fixed["Month"].mean()
month_std = full_fixed["Month"].std()
age_mean = full_fixed["Track Age"].mean()
age_std = full_fixed["Track Age"].std()

# keep the countries and complete years from the full dataset
country_list = sorted(full_data["Country"].unique())
year_index = full_data["Month"] // 12
year_count = full_data.groupby(year_index)["Month"].nunique()
time_levels = sorted(year_count[year_count == 12].index)
year_list = [2019 + int(level) for level in time_levels]

"""
Running the model:
"""

# prior hyperparameters
numerical_mean = 0.0
numerical_sd = 0.5
dummy_mean = 0.0
dummy_sd = 0.5

# observation-variance settings
pilot_variance = 0.2
mixture_convergence_tolerance = 1.0e-8
track_count_pilot = 5000
tau_start_by_country = {"Argentina": 2.7196039363479945}

output_dir = os.path.dirname(__file__)
cache_root_dir = os.path.join(output_dir, "model-cache-country")
slope_rows = []

print("Running model...")

# fit the model separately for every country and keep the fitted slopes in memory
for country in tqdm(country_list, desc="Countries", unit="country"):

    # keep only the current country rows
    data = full_data[full_data["Country"] == country].copy()

    # sort the song ids and assign ascending integers
    song_lookup = pd.DataFrame({"Song ID": np.sort(data["Song ID"].unique())})
    song_lookup["Song Num"] = np.arange(1, len(song_lookup) + 1)
    data = data.merge(song_lookup, on="Song ID", how="left")

    # define fixed effects
    fixed_effects = data[["Month", "Track Age"]]

    # standardise the fixed effects on the full-data scale
    fixed_effects["Month"] = (fixed_effects["Month"] - month_mean) / month_std
    fixed_effects["Track Age"] = (fixed_effects["Track Age"] - age_mean) / age_std

    # register groups for interaction expansion across the existing columns
    groups = {}
    groups["Month"] = ["Month"]
    groups["Track Age"] = ["Track Age"]

    # add interaction terms
    terms = [("Track Age", "Month")]
    fixed_effects = interaction_table(fixed_effects, groups, terms)

    # keep the fitted design structure so the pilot uses the same columns
    fixed_cols = fixed_effects.columns
    interaction_terms = fixed_effects.attrs["interaction_terms"]

    # build the fixed-effect prior precision for the current country design
    fixed_precision_prior = prior_vector(fixed_cols, numerical_sd, dummy_sd)

    # instantiate GLMM and prior
    linear_model = GLMM(tau_start=tau_start_by_country.get(country, np.exp(4.0)))
    prior = gaussian_prior(fixed_precision_prior)

    # setup random effects
    group_index, rand_count = linear_model.prepare_groups(data["Song Num"])
    random_effects = pd.Series(group_index, index=data.index)

    # define the log1p-transformed response variable
    response = np.log1p(data["Streams"])

    # limit the pilot size to the songs available for the current country
    pilot_count = min(track_count_pilot, data["Song Num"].nunique())

    # sample the pilot rows from the dataset
    pilot_data = random_sample(data, pilot_count)

    # rebuild the pilot fixed-effects design on the full-data scale and columns
    pilot_fixed_effects = fixed_table(pilot_data, month_mean, month_std, age_mean, age_std, terms, fixed_cols, interaction_terms)

    # build pilot random intercepts from the pilot song identifiers
    pilot_linear_model = GLMM()
    pilot_group_index, pilot_rand_count = pilot_linear_model.prepare_groups(pilot_data["Song Num"])
    pilot_random_effects = pd.Series(pilot_group_index, index=pilot_data.index)

    # transform the pilot response onto the modelled log1p scale
    pilot_response = np.log1p(pilot_data["Streams"])

    # fit the pilot model with the original fixed observation variance
    pilot_likelihood_model = GaussianLog1pLikelihood(variance=pilot_variance)
    pilot_inla = inlaSolver(pilot_linear_model, pilot_likelihood_model, prior, 0.0, fixed_precision_prior, pilot_fixed_effects, pilot_response, pilot_random_effects)

    # run the pilot fit so the pilot posterior means are available for residual calculation
    (pilot_intermean, pilot_intervar, pilot_fixedmeans, pilot_fixedvars, pilot_fixedcov, pilot_randmeans,
     pilot_randvars, pilot_tau_candidates, pilot_tau_weights, pilot_fixed_mix, pilot_rand_mix,
     pilot_response_params) = pilot_inla.inla(pilot_fixed_effects, pilot_response, pilot_random_effects)

    # convert the pilot design and pilot random ids into numpy arrays for indexing
    pilot_fixed_np = pilot_fixed_effects.to_numpy(dtype=float)
    pilot_rand_np = pilot_random_effects.to_numpy()

    # combine pilot intercept, fixed-effect and random-effect contributions
    pilot_fitted = float(pilot_intermean) + (pilot_fixed_np @ np.asarray(pilot_fixedmeans, dtype=float)) + np.asarray(pilot_randmeans, dtype=float)[pilot_rand_np]

    # estimate the fixed mixed Gaussian parameters from the demeaned pilot residuals
    pilot_residual = demeaned_residuals(pilot_response, pilot_fitted, pilot_data["Song Num"])
    likelihood_model = MixedGaussianLikelihood(residual_vals=pilot_residual, convergence_tolerance=mixture_convergence_tolerance)

    # instantiate INLA solver
    inla = inlaSolver(linear_model, likelihood_model, prior, 0.0, fixed_precision_prior, fixed_effects, response, random_effects)

    # point the solver at this country's cache file inside the shared cache folder
    inla.model_cache_path = os.path.join(cache_root_dir, f"inla-cache_{country}.pkl")
    inla.model_cache_dir = os.path.dirname(inla.model_cache_path)

    # avoid loading another country's cache when this country has not been cached yet
    if os.path.isfile(inla.model_cache_path):
        os.utime(inla.model_cache_path, None)
    else:
        inla.load_inla_cache = lambda: False

    # run the model and get the results
    fixed_results, rand_results, precision_results = inla.results()

    # keep the fitted age slope for each complete year
    coef_vals = fixed_results["means"]
    fixed_cov = np.asarray(inla.posterior_fixedcov, dtype=float)
    cov_names = ["(Intercept)"] + list(fixed_cols)
    age_idx = cov_names.index("Track Age")
    age_month_idx = cov_names.index("Track Age*Month") if "Track Age*Month" in cov_names else None
    ci_z = 1.959963984540054

    for level, year in zip(time_levels, year_list):
        month_z = ((float(level) * 12.0) - month_mean) / month_std
        age_slope = float(coef_vals["Track Age"])
        slope_weights = np.zeros(len(cov_names), dtype=float)
        slope_weights[age_idx] = 1.0

        if "Track Age*Month" in coef_vals.index:
            age_slope += float(coef_vals["Track Age*Month"]) * month_z
            slope_weights[age_month_idx] = month_z

        age_slope_var = float(slope_weights @ fixed_cov @ slope_weights)
        age_slope_sd = float(np.sqrt(max(age_slope_var, 0.0)))
        age_slope_lower = age_slope - (ci_z * age_slope_sd)
        age_slope_upper = age_slope + (ci_z * age_slope_sd)

        slope_rows.append({"Country": country, "Year": year, "age_slope": age_slope,
                           "age_slope_lower": age_slope_lower, "age_slope_upper": age_slope_upper})

# turn the stored slope rows into one plotting table
slope_data = pd.DataFrame(slope_rows)

# display the collated country graphs
plot_results(slope_data, output_dir)

end = time.perf_counter()
elapsed = end - start
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)
print(f"Total computation time: {hours}:{minutes}:{seconds}")
