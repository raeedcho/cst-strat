---
# weight: 2021120901
title: CO-CST covariance analysis
date: "2021-12-09"
tags:
- CO/CST comparison
- neural
- population
- covariance
- dimensionality
---

For neural covariance analysis, we have a number of possible questions:

- :question: Because CST has smaller movements, does neural variance also decrease as much as you would expect?
    - :bar_chart: Analysis: compare total neural variance between CO and CST
    - :heavy_check_mark: CST usually has smaller variance than CO, but unclear yet if it's what we would expect from smaller movements
    - :question: Remiaining question: is this variance solely due to smaller movements? Not sure how to address this question though.
- :question: CST seems like a more complex task than CO--does this mean that neural activity explores more dimensions?
    - :bar_chart: compare CO and CST covariance rank (i.e. dimensionality), using participation ratio, parallel analysis, and GPFA cross-validated log-likelihood
        - :heavy_check_mark: CST does not seem to have much higher dimensionality, but activity is much closer to noise floor
        - :speech_balloon: Three methods of dimensionality estimation (participation ratio, parallel analysis, and GPFA cross-validated log-likelihood) give differing results
- :question: How much do CST and CO overlap in the neural manifold?
    - :bar_chart: compare neural covariance dimensions between CO and CST, using subspace overlap or principle angles

I examined the first two questions together, during the movement period of both CO and CST (figures below). Important to note: I calculated all of these after softnorming neural activity within each task ($\alpha$=5). No smoothing applied. Applying these techniques without the softnorm gives similar results for variance, though parallel analysis results are noisier (probably due to high variability of a few neurons). Reasonable levels of smoothing doesn't change any of these results much, though I would guess that oversmoothing data would decrease dimensionality.

![Earl Movement variance/dimensionality comparison](figs/softnormed/20211203_Earl20190716_COCSTmove_vardim_comparison-01.png)

![Ford Movement variance/dimensionality comparison](figs/softnormed/20211203_Ford20180627_COCSTmove_vardim_comparison-01.png)

## Neural Variance

Despite the within-task softnorm, the total neural variance seems much lower for CST than CO (top plots, CO in red, CST in black). Variance for individual lambdas of the CST (solid line) doesn't seem to be much different from the variance over all lambdas (black dashed line). Remaining question: is this difference in variance more or less than what we would expect from the different kinematics of the two tasks? CST movements are much smaller than CO movements on average, so we might expect a much lower neural variance to start with.

## Neural dimensionality

The same plots above show two measures of dimensionality: the participation ratio, which a measure of how flat the eigenspectrum of a covariance matrix is (flat eigenspectrum would have PR equal to the number of neurons), and parallel analysis dimensionality, which constructs a null distribution of eigenspectra by shuffling neural activity independently across time. The upshot is that PR says that the eigenspectra of both tasks are remarkably flat (when neurons are softnormed at least), but PA suggests that both tasks have significant covariance in only a few dimensions. Notably, when evaluated at individual $\lambda$, the PA dimensionality (black solid line) is very similar to the CO dimensionality (red dashed line), while CST dimensionality when evaluated over all $\lambda$ (black dashed line) is notably higher (though unclear if this difference is statistically significant). The most mundane explanation for this difference is that when you add more data, you get higher estimated dimensionality (see Surya Ganguli's work and Ryan Williamson's work). A less mundane explanation might be that you use different, not fully overlapping dimensions of neural activity at different $\lambda$, but this claim seems unlikely.

Below are some eigenspectra for CO and CST movement-time activity for both monkeys, with 95th percentile null distribution from parallel analysis.

![Earl CO movement-time eigenspectrum](figs/softnormed/20211203_Earl20190716_COmove_eigenspectrum_with_PA-01.png)

![Earl CST movement-time eigenspectrum](figs/softnormed/20211203_Earl20190716_CSTmove_eigenspectrum_with_PA-01.png)

![Ford CO movement-time eigenspectrum](figs/softnormed/20211203_Ford20180627_COmove_eigenspectrum_with_PA-01.png)

![Ford CST movement-time eigenspectrum](figs/softnormed/20211203_Ford20180627_CSTmove_eigenspectrum_with_PA-01.png)

### GPFA dimensionality estimation

We can also evaluate the dimensionality using GPFA cross-validated log-likelihood across number of factors used to describe the data. For this dataset, it has been a little tricky to fairly compare CO and CST, since generally there's so much more CST data than CO. But just pushing the data through the pipeline, we get these figures (note that these data were not softnormed before running GPFA, and this particular analysis takes hours to run):

![Ford CO GPFA cross-validated log-likelihood](figs/non-normed/gpfa/Ford_20180627_CO_gpfa_LLvDim-01.png)

![Ford CST GPFA cross-validated log-likelihood](figs/non-normed/gpfa/Ford_20180627_CST_full_gpfa_LLvDim-01.png)

This suggests that CST does indeed have higher dimensionality than CO. However, when we select only the number of CST trials (each 6 seconds long) to match the total number of time points across tasks, we get this for CST:

![Ford CST GPFA cross-validated log-likelihood](figs/non-normed/gpfa/Ford_20180627_CST_timepointmatch_gpfa_LLvDim-01.png)

And that seems to suggest a similar dimensionality between the two tasks.

But we can also match both number of trials and number of timepoints in each tasks, subselecting one 0.5 s portion of each CST trial to match the CO trials:

![Ford CST trial-matched GPFA cross-validated log-likelihood](figs/non-normed/gpfa/Ford_20180627_CST_trialmatch_gpfa_LLvDim-01.png)

And that gives something mostly nonsensical. Haven't quite figured this one out yet, but I think the overall story is just that CST and CO have similar dimensionality, and everything weird here is just measurement error.

### Hold time analyses

I also ran all of these analyses (except the GPFA analysis) on the hold time before movement in both tasks. Here are those figures:

![Earl hold variance/dimensionality comparison](figs/softnormed/20211203_Earl20190716_COCSThold_vardim_comparison-01.png)

![Ford hold variance/dimensionality comparison](figs/softnormed/20211203_Ford20180627_COCSThold_vardim_comparison-01.png)

![Earl CO hold-time eigenspectrum](figs/softnormed/20211203_Earl20190716_COhold_eigenspectrum_with_PA-01.png)

![Earl CST hold-time eigenspectrum](figs/softnormed/20211203_Earl20190716_CSThold_eigenspectrum_with_PA-01.png)

![Ford CO hold-time eigenspectrum](figs/softnormed/20211203_Ford20180627_COhold_eigenspectrum_with_PA-01.png)

![Ford CST hold-time eigenspectrum](figs/softnormed/20211203_Ford20180627_CSThold_eigenspectrum_with_PA-01.png)

The scales are a little bit off (note that variance y-axis has quite tiny range compared to previous figures), but basically, variance and dimensionality were pretty much the same in the hold period of CO and CST. Also, Ford's neural data during hold period is quite noisy, as shown by his hold-time eigenspectra being close to the noise.
