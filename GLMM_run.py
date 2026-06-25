# -*- coding: utf-8 -*-
"""
Runs the model using the OOP-based model elements (use Bayesian_GLMM if you want a straight GLMM, as it's faster due to less object passing)
This version is for when you want to change the model, likelihood and prior.
However, bear in mind, the INLA solver will only work (currently) with latent Gaussian models based on finite hyperparameters and sparse precision
For example:
    Most generalised linear models (fixed-effect, mixed-effect...etc...)
    Generalised additive models
    Mixed models with multiple random-effect terms and random slopes
    Time-series latent models with autoregressive and random-walk effects
    Spatial lattice models
    Spatial continuous-field models
    Survival models

To run this:
    instantiate the linear model and likelihood model
    set up the random effects (if there are any) using the linear model
    instantiate the inla, 
        passing it the model components: the linear model, likelihood model and prior; 
        then the data: fixed effects (dataframe), response (dataframe) and random effects (dataframe - if exists)
    call the results

@author: Chris
"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker
import itertools

from GLMM_rand import GLMM
from INLA_solver import inlaSolver
from gaussian_log1p import GaussianLog1pLikelihood
from gaussian_mixed import MixedGaussianLikelihood
# from priors import uniform

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

# plot the slopes for the results - need to consider this better later - use a paper example, as I'm not sure the slopes are what I need here
def plot_results(fixed_vals, fixed_results, data):

    font_size = 14

    # get level lists
    country_list = sorted(data["Country"].unique())

    # get the complete year index levels
    year_index = data["Month"] // 12
    year_count = data.groupby(year_index)["Month"].nunique()
    time_levels = sorted(year_count[year_count == 12].index)
    time_labels = np.arange(1, len(time_levels) + 1)

    # build the grid
    country_df = pd.DataFrame({"Country": country_list})
    time_df = pd.DataFrame({"Year": time_levels})

    slope_data = country_df.merge(time_df, how="cross")

    # put year values on the same scale as the fitted Month column
    month_mean = data["Month"].mean()
    month_std = data["Month"].std()
    month_vals = slope_data["Year"].to_numpy() * 12.0
    slope_data["Month_z"] = (month_vals - month_mean) / month_std

    # compute the country-year Track Age coefficient directly from the interaction terms
    coef_vals = fixed_results["means"]
    slope_data["age_slope"] = float(coef_vals["Track Age"])

    if "Track Age*Month" in coef_vals.index:
        slope_data["age_slope"] += float(coef_vals["Track Age*Month"]) * slope_data["Month_z"]

    for country in country_list:
        country_mask = slope_data["Country"].astype(str) == str(country)

        age_country_col = f"Track Age*Country{country}"
        if age_country_col in coef_vals.index:
            slope_data.loc[country_mask, "age_slope"] += float(coef_vals[age_country_col])

        age_month_country_col = f"Track Age*Month*Country{country}"
        if age_month_country_col in coef_vals.index:
            slope_data.loc[country_mask, "age_slope"] += (
                float(coef_vals[age_month_country_col]) * slope_data.loc[country_mask, "Month_z"]
            )

    # reshape the plotted slopes into a country-by-year table
    slope_export = slope_data[["Country", "Year", "age_slope"]].copy()
    slope_export["Year"] = 2019 + slope_export["Year"].astype(int)
    slope_export = slope_export.pivot(index="Country", columns="Year", values="age_slope").reset_index()
    slope_export.columns = ["Country"] + [str(year) for year in slope_export.columns[1:]]

    # save the plotted slopes in the same table layout used for the figure
    slope_export.round(3).to_csv(os.path.join(os.path.dirname(__file__), "country_slopes.csv"), index=False)

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
    axis.set_xticklabels([2019 + int(level) for level in time_levels])
    axis.set_xlim(time_labels[0], time_labels[-1])
    axis.tick_params(axis="both", labelsize=font_size)

    # housekeeping
    plt.tight_layout()
    figure.savefig(os.path.join(os.path.dirname(__file__), "plot_results.png"), dpi=300, bbox_inches="tight")
    plt.show()

    return

# plot change in the average age predicted by the model against the first complete year
def plot_change(fixed_vals, fixed_results, data):

    font_size = 14

    # get level lists
    country_list = sorted(data["Country"].unique())

    # get the complete year index levels
    year_index = data["Month"] // 12
    year_count = data.groupby(year_index)["Month"].nunique()
    time_levels = sorted(year_count[year_count == 12].index)
    time_labels = np.arange(1, len(time_levels) + 1)

    # get the fixed-effect coefficients used for fitted values
    intercept_val = fixed_results.loc["(Intercept)", "means"]
    coef_vals = fixed_results.loc[fixed_vals.columns, "means"].to_numpy()

    # predict streams on the observed rows using the fixed effects
    fitted_streams = np.expm1(intercept_val + (fixed_vals.to_numpy(dtype=float) @ coef_vals))

    # collect the rows needed for country-year age averages
    change_data = data[["Country", "Track Age", "Month"]].copy()
    change_data["Year"] = change_data["Month"] // 12
    change_data["fitted_streams"] = fitted_streams
    change_data = change_data[change_data["Year"].isin(time_levels)].copy()

    # calculate the weighted average age for each country and year
    change_rows = []
    for country in country_list:
        country_data = change_data[change_data["Country"] == country]
        for year in time_levels:
            year_data = country_data[country_data["Year"] == year]
            weighted_age = float(np.sum(year_data["Track Age"] * year_data["fitted_streams"]) / np.sum(year_data["fitted_streams"]))
            change_rows.append({"Country": country, "Year": year, "change": weighted_age})

    # turn the collected rows into a plotting table
    change_data = pd.DataFrame(change_rows)

    # subtract each country's first-year value so every line starts at zero
    start_year = time_levels[0]
    start_change = change_data[change_data["Year"] == start_year][["Country", "change"]].rename(columns={"change": "start_change"})
    change_data = change_data.merge(start_change, on="Country", how="left")
    change_data["change"] = change_data["change"] - change_data["start_change"]

    # create figure and axes
    figure, axis = plt.subplots(figsize=(12, 12))
    colour_vals = muted_colours(len(country_list))

    # draw one line per country and store its final point
    line_ends = []
    for index, country in enumerate(country_list):
        country_plot_data = change_data[change_data["Country"] == country].sort_values("Year")
        change_vals = country_plot_data["change"].to_numpy(dtype=float)
        axis.plot(time_labels, change_vals, color=colour_vals[index], linewidth=1.8)
        line_ends.append({"country": country, "y": change_vals[-1], "colour": colour_vals[index]})

    # draw a zero line for the first-year reference
    axis.axhline(0.0, color="0.75", linewidth=1.0)

    # draw the country labels to the right of the line ends
    label_x = time_labels[-1] + 0.15
    label_ends(axis, line_ends, time_labels[-1], label_x, font_size=font_size)

    # force x ticks onto evenly-spaced year positions
    axis.set_xticks(time_labels)
    axis.set_xticklabels([2019 + int(level) for level in time_levels])
    axis.set_xlim(time_labels[0], time_labels[-1])
    axis.tick_params(axis="both", labelsize=font_size)

    # keep the scientific notation offset label the same size as the tick labels
    axis.yaxis.get_offset_text().set_fontsize(font_size)

    # housekeeping
    plt.tight_layout()
    figure.savefig(os.path.join(os.path.dirname(__file__), "plot_change.png"), dpi=300, bbox_inches="tight")
    plt.show()

    return

# plot observed and fitted average age globally, then extend the fitted line into the future
def plot_future(fixed_vals, fixed_results, rand_results, data, month_mean, month_std, age_mean, age_std,
                country_cols, terms, fixed_cols, interaction_terms, future_end_year=2030):

    font_size = 14

    # get the complete year index levels
    year_index = data["Month"] // 12
    year_count = data.groupby(year_index)["Month"].nunique()
    time_levels = sorted(year_count[year_count == 12].index)
    calendar_years = np.array([2019 + int(level) for level in time_levels], dtype=int)

    # collect the observed rows used in the yearly summaries
    plot_data = data[["Song Num", "Track Age", "Month", "Streams", "Country"]].copy()
    plot_data["Year"] = plot_data["Month"] // 12
    plot_data = plot_data[plot_data["Year"].isin(time_levels)].copy()

    # combine the fixed and random effects to get fitted streams on the observed rows
    intercept_val = fixed_results.loc["(Intercept)", "means"]
    coef_vals = fixed_results.loc[fixed_vals.columns, "means"].to_numpy()
    rand_means = rand_results["means"].loc[data["Song Num"]].to_numpy(dtype=float)
    fitted_streams = np.expm1(intercept_val + (fixed_vals.to_numpy(dtype=float) @ coef_vals) + rand_means)
    plot_data["fitted_streams"] = fitted_streams[plot_data.index]

    # calculate the observed and fitted average age for each observed year
    observed_age = []
    fitted_age = []
    for year in time_levels:
        year_data = plot_data[plot_data["Year"] == year]
        observed_age.append(float(np.sum(year_data["Track Age"] * year_data["Streams"]) / np.sum(year_data["Streams"])))
        fitted_age.append(float(np.sum(year_data["Track Age"] * year_data["fitted_streams"]) / np.sum(year_data["fitted_streams"])))

    # keep the final observed year as the template for the future projection
    last_year = time_levels[-1]
    last_calendar_year = int(calendar_years[-1])
    template_data = plot_data[plot_data["Year"] == last_year][["Song Num", "Country", "Track Age", "Month"]].copy()

    # project the fitted average age forward by ageing the final-year rows and moving them through time
    future_rows = []
    for calendar_year in range(last_calendar_year + 1, future_end_year + 1):
        year_shift = calendar_year - last_calendar_year
        future_data = template_data.copy()
        future_data["Track Age"] = future_data["Track Age"] + year_shift
        future_data["Month"] = future_data["Month"] + (12 * year_shift)
        future_data["Year"] = future_data["Month"] // 12

        # rebuild the future design on the fitted scale and predict fitted streams for those rows
        future_fixed = fixed_table(future_data, month_mean, month_std, age_mean, age_std,
                                   country_cols, terms, fixed_cols, interaction_terms)
        future_rand = rand_results["means"].loc[future_data["Song Num"]].to_numpy(dtype=float)
        future_streams = np.expm1(intercept_val + (future_fixed.to_numpy(dtype=float) @ coef_vals) + future_rand)

        # store the projected weighted average age for that future year
        projected_age = float(np.sum(future_data["Track Age"] * future_streams) / np.sum(future_streams))
        future_rows.append({"CalendarYear": calendar_year, "age": projected_age})

    # collect the projected line so it continues directly from the last fitted point
    projected_years = np.array([last_calendar_year] + [row["CalendarYear"] for row in future_rows], dtype=int)
    projected_age = np.array([fitted_age[-1]] + [row["age"] for row in future_rows], dtype=float)

    # create the figure and plot the three global lines
    figure, axis = plt.subplots(figsize=(12, 12))
    axis.plot(calendar_years, observed_age, color="0.25", linewidth=2.0, label="Observed")
    axis.plot(calendar_years, fitted_age, color="#2f6f9f", linewidth=2.0, label="Fitted")
    axis.plot(projected_years, projected_age, color="#2f6f9f", linewidth=2.0, linestyle="--", label="Projected")

    # force year ticks onto the observed and projected calendar years
    axis.set_xticks(np.arange(int(calendar_years[0]), future_end_year + 1))
    axis.set_xlim(int(calendar_years[0]), future_end_year)
    axis.tick_params(axis="both", labelsize=font_size)

    # keep the scientific notation offset label the same size as the tick labels
    axis.yaxis.get_offset_text().set_fontsize(font_size)

    # add a simple legend for the three global series
    axis.legend(frameon=False, fontsize=font_size)

    # housekeeping
    plt.tight_layout()
    figure.savefig(os.path.join(os.path.dirname(__file__), "plot_future.png"), dpi=300, bbox_inches="tight")
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

            # record interaction terms metadata
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
fixed effects: day_index, age_days, territory and genre 
random effect (groupings): song_id - MUST be an ascending integer for the above model to work
interaction term: age*time, age*territory, age*time*territory

"""
start = time.perf_counter()

print("Loading data...")

# load in the model data
# data = pd.read_csv("../Data/main_data_complete_30_days.csv",
data = pd.read_csv("../Data/main_data_complete_cyear.csv",
                   usecols=["Country", "Luminate Song ID", "Track Age", "Month", "Streams"])

# rename the model columns
data = data.rename(columns={"Luminate Song ID": "Song ID"})

# remove rows with missing model data
data = data.dropna(subset=["Country", "Song ID", "Track Age", "Month", "Streams"])

# remove rows with negative track age
data = data[data["Track Age"] >= 0]

# sort the song ids and assign ascending integers
song_lookup = pd.DataFrame({"Song ID": np.sort(data["Song ID"].unique())})
song_lookup["Song Num"] = np.arange(1, len(song_lookup) + 1)
data = data.merge(song_lookup, on="Song ID", how="left")

# take a random sample
# data = random_sample(data, 10000)

# define fixed effects
fixed_effects = data[["Month", "Track Age"]]

# dummy encode the country categories with Booleans
dummy_country = pd.get_dummies(data["Country"], drop_first=True, prefix="Country", prefix_sep="").astype(int)
fixed_effects = fixed_effects.join(dummy_country)

# keep the numerical scaling used by the fitted design
month_mean = fixed_effects["Month"].mean()
month_std = fixed_effects["Month"].std()
age_mean = fixed_effects["Track Age"].mean()
age_std = fixed_effects["Track Age"].std()

# standardise the fixed effects (not the dummies though)
fixed_effects["Month"] = (fixed_effects["Month"] - month_mean) / month_std
fixed_effects["Track Age"] = (fixed_effects["Track Age"] - age_mean) / age_std

# keep the country columns from the fitted design
country_cols = list(dummy_country.columns)

# register groups for interaction expansion across the existing columns
groups = {}
groups["Month"] = ["Month"]
groups["Track Age"] = ["Track Age"]
groups["Country"] = country_cols

# add interaction terms
terms = [("Track Age", "Month"),
         # ("Month", "Country"),
         ("Track Age", "Country"),
         ("Track Age", "Month", "Country")]
fixed_effects = interaction_table(fixed_effects, groups, terms) # toggle on and off to add the interaction term to the model

# keep the fitted design structure so the pilot uses the same columns
fixed_cols = fixed_effects.columns
interaction_terms = fixed_effects.attrs["interaction_terms"]

# print("fixed effect NaNs: ", fixed_effects.isna().sum()) # sanity check

"""
Running the model:
"""

# prior hyperparameters
numerical_mean = 0.0
numerical_sd = 0.5 # baseline == 0.5 - set to 1.0 to test the effect of the prior on the posterior
dummy_mean = 0.0
dummy_sd = 0.5
fixed_precision_prior = prior_vector(fixed_cols, numerical_sd, dummy_sd)

# observation-variance settings
pilot_variance = 0.2 # initial variance as pilot fits observation variance
mixture_convergence_tolerance = 1.0e-8 # convergence threshold for fitting the pilot residual Gaussian mixture
track_count_pilot = 1000 # the data sample sized used to derive the variance - e.g. 1000 tracks

# instantiate GLMM and prior
linear_model = GLMM()
prior = gaussian_prior(fixed_precision_prior)
# prior = uniform(likelihood_model)

# setup random effects
group_index, rand_count = linear_model.prepare_groups(data["Song Num"])
random_effects = pd.Series(group_index, index=data.index)

# define the log1p-transformed response variable
response = np.log1p(data["Streams"])

# sample the pilot rows from the dataset
pilot_data = random_sample(data, track_count_pilot)

# report the pilot size before fitting the pilot model
print("Running pilot for the mixed Gaussian fit...")

# rebuild the pilot fixed-effects design on the exact final design scale and columns
pilot_fixed_effects = fixed_table(pilot_data, month_mean, month_std, age_mean, age_std, country_cols, terms, fixed_cols, interaction_terms)

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

# combine pilot intercept, fixed-effect, and random-effect contributions
pilot_fitted = (float(pilot_intermean) + (pilot_fixed_np @ np.asarray(pilot_fixedmeans, dtype=float)) + np.asarray(pilot_randmeans, dtype=float)[pilot_rand_np])

# estimate the fixed mixed Gaussian parameters from the demeaned pilot residuals
pilot_residual = demeaned_residuals(pilot_response, pilot_fitted, pilot_data["Song Num"])
likelihood_model = MixedGaussianLikelihood(residual_vals=pilot_residual,convergence_tolerance=mixture_convergence_tolerance)
print(f"Narrow component weight: {likelihood_model.mixture_weight:.4f}")
print(f"Narrow component standard deviation: {np.sqrt(likelihood_model.narrow_variance):.4f}")
print(f"Wide component standard deviation: {np.sqrt(likelihood_model.wide_variance):.4f}")

# instantiate INLA solver 
inla = inlaSolver(linear_model, likelihood_model, prior, 0.0, fixed_precision_prior, fixed_effects, response, random_effects)

# use a temp folder for the cache - deactivate to save back to default
output_dir = os.path.dirname(__file__)
inla.model_cache_dir = os.path.join(output_dir, "model-cache")
inla.model_cache_path = os.path.join(inla.model_cache_dir, "inla-cache.pkl")

# run the model and get the results
fixed_results, rand_results, precision_results = inla.results()

# save the model results
fixed_results.to_csv(os.path.join(output_dir, "fixed_effects.csv"))
rand_results.to_csv(os.path.join(output_dir, "random_effects.csv"))
precision_results.to_csv(os.path.join(output_dir, "hyperparameters.csv"))

# run the posterior checks
inla.posterior()

# save the pointwise Pareto-k values from PSIS-LOO
pareto_k_results = pd.DataFrame({"pareto_k": inla.pareto_k}, index=response.index)
pareto_k_results.to_csv(os.path.join(output_dir, "pareto_k.csv"))

# display the model graphs
plot_results(fixed_effects, fixed_results, data)
# plot_change(fixed_effects, fixed_results, data) # basically useless this one but worked too hard on it to just delete it
# plot_future(fixed_effects, fixed_results, rand_results, data, month_mean, month_std, age_mean, age_std, country_cols, terms, fixed_cols, interaction_terms) # doesn't really work due to the log1p link, which doesn't scale when exponentiated back to raw streams - didn't use in the end

end = time.perf_counter()
elapsed = end - start
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)
print(f"Total computation time: {hours}:{minutes}:{seconds}")
