---
title: "Switching neural dynamical system analysis"
date: 2021-12-18T15:42:20-05:00
draft: true
katex: true
tags:
- neural
- population
- dynamics
- switching
---

In the CO behavior, we know that the epoch when a monkey executes a movement has significantly different neural dynamics from the epoch in which a monkey prepares the movement. In particular, preparatory dynamics seem to draw the neural state towards an initial condition fixed point from which (on the go cue), the neural state then dynamically evolves to execute the movements. [^1]

[^1]: Side note: recent work suggests that this switching dynamics occurs through a cortico-thalamo-cortical loop mediated by the basal ganglia. See Kao et al. 2021 and Logiaco et al. 2021 for some computational models

Because we see something that looks like intermittent movements in the CST behavior, it may be the case that neural dynamics also switch between different states during CST. To examine this, I modeled neural activity using a switching linear dynamical system (SLDS), as described by Linderman et al. 2016.

## Brief description of SLDS model

For this analysis, I modeled neural activity as follows:

$$
z_{t+1} ~ Cat(K,\pi(z_t))
x_{t+1} ~ Normal(A_{z_t} x_t, \Sigma_{z_t})
y_{t} ~ Poisson(C x_t)
$$

Here, $z$ is a hidden discrete state, taking on one of $K$ values, where the probability of transition to any other value depends on only the current state, with the transition probability given by $\pi(z_t)$. On the next level, $x$ comprises a hidden continuous state (which could be a vector). The evolution of this continuous state follows a typical linear dynamical system with Gaussian noise, with the caveat that the dynamics matrix and the noise characteristics depend on the current hidden discrete state, $z$. At the final, observable level, $y$ is the observed emissions from this model, the number of neural spikes in a given time bin. $y$ is a projection of the hidden continuous state $x$ through emissions matrix $C$.

In fitting this model, we infer the matrices $C$, $A_{z}$ (for each of the $K$ possible discrete state values), $\Sigma_{z}$ (for each of the $K$ possible values of $z$). Simultaneously, we also infer the expected values of $z_t$ and $x_t$ over all time points [^2]. Consequently, once we perform the inference of these hidden states, we have a low-dimensional representation of the neural activity that evolves according to simple dynamical rules.

[^2]: In the simplest cases, we can fit this using expectation-maximization, which computes the expected values of the hidden states, and then calculates the maximum likelihood estimates of the parameter matrices. This process is repeated iteratively until convergence. For this particular model, we can't use this exact procedure, but the idea is still there. Instead we fit this using a clever approximation of expectation-maximization--however, I'm not an expert in this math, so I won't be going through the specifics.

Before fitting, we also have to specify a number of hyperparameters. These include the number of possible discrete state values ($K$) and the dimensionality of the continuous state $x$. We can also change the emissions model (Poisson above), depending on how we process our neural data.

## Applying the SLDS to CO data

As a proof of concept, we can apply this SLDS model to the CO data, where we know there should be a transition in neural dynamics between waiting in the center and moving towards the outer target. Below is a figure showing the inferred discrete and continuous hidden states from an example CO trial. For reference the number of discrete states was 2 (which would ideally correspond to "hold" and "move") and the dimensionality of the continuous state was 8 [^3].

[^3]: I need to check this to make sure...

Earl CO SLDS inference example trial figure
<!--![Earl CO SLDS inference example trial](figs/trials/co/example.png)-->

In this figure, the top plot shows the kinematics of the hand, the second plot shows the inferred value of the discrete state (by color), the third plot shows the values of all dimensions of the continuous latents, and the bottom plot shows a raster of neural activity. In all plots, the red dashed line corresponds to the go cue. Note that neither the behavior nor the timing of the go cue were used to fit the model--the SLDS infers the parameters and hidden states only from neural activity. As expected, the SLDS generally does a good job of figuring out that there is one switch in dynamics in this trial, and it tends to happen around the time of the go cue [^todo]. 

[^todo]: Yet to be done: I need to quantify how well this switching time matches up with movement onset, or something of the sort, just to validate that this inference is useful.

Here's a slideshow of CO trials:

_CO trial slideshow_

## Applying the SLDS to CST data

Now that we know the SLDS at least sort of works on CO data, we can apply it to CST data. Unfortunately, the results don't seem to be quite as interpretable for CST, at least at first glance. Here's an example trial.

_Earl CST SLDS inference example trial figure_

Here's a slideshow of CST trials from this model run:

_CST trial slideshow_

### Tweaks to the SLDS model

There are a number of tweaks we can make to the model to perhaps make this result cleaner:

- Change the number of possible discrete states
- Make the transitions between discrete states "sticky"--i.e. upweight the prior probability that when $z$ is in a particular state, it'll stay in that state
- Increase bin size and change the emissions model from Poisson to Gaussian
