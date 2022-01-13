---
title: Analysis brainstorming
weight: 1
date: 2021-12-01
layout: "single"
tags:
---

Below is a broad list of all the analyses that we might run on CST data. The main focus is on neural analyses, which themselves are primarily focused on comparing CST and CO data, but I've also included a short section on behavioral analyses that could help us understand and contextualize the neural analysis results.

If you want to see a log of some of the analyses that I've run on the data, check out the [Analysis log](analysis-log/)!

If you have some ideas on analyses that aren't listed below, leave a comment at the bottom of the article (requires GitHub account), or send me an email.

## Neural analyses

### Single neuron analyses

List of analyses that we can use to compare CO and CST on a single neuron level

- Averages
    - :question: How does average firing rate change across tasks?
        - :bar_chart: Analysis: compare averages across tasks
    - :question: How does average firing rate change between hold and movement, within and across tasks?
        - :bar_chart: Analysis: Hold vs. move activity within and across tasks
    - :question: Can we see a neural engagement signal in the average rates of neurons right after a $\lambda$ change?
        - :bar_chart: Analysis: look at average neural activity in movement period of CST per trial, plotted as a function of # of trials after $\lambda$ change
- PSTHs
    - :question: Do PSTHs in CST look like CO?
        - :bar_chart: Analysis: Calculate PSTHs for hold->move submovements in CST
- Tuning curves
    - :question: How similar is single neural tuning between CO and CST?
        - :bar_chart: Analysis: Compare left/right selectivity between CO and CST
            - :heavy_check_mark: Note: This is basically the sensorimotor index analysis, which showed similar tuning between the tasks
        - :bar_chart: Analysis: Fit GLMs and examine how parameters shift between CO and CST
            - :exclamation: Note/caution: neural GLMs are usually noisy and tricky to glean interpretation from

### Population analyses

List of [population analysis](tags/population) methods we could use to compare CO and CST

- Neural [covariance](tags/covariance)
    - :question: Because CST has smaller movements, does neural variance also decrease as much as you would expect?
        - :bar_chart: Analysis: compare total neural variance between CO and CST
        - :heavy_check_mark: CST usually has smaller variance than CO, but unclear yet if it's what we would expect from smaller movements
        - :question: Remiaining question: is this variance solely due to smaller movements? Not sure how to address this question though.
    - :question: CST seems like a more complex task than CO--does this mean that neural activity explores more dimensions?
        - :bar_chart: compare CO and CST covariance rank (i.e. dimensionality)
            - :heavy_check_mark: CST does not seem to have much higher dimensionality, but activity is much closer to noise floor
            - :speech_balloon: Three methods of dimensionality estimation (participation ratio, parallel analysis, and GPFA cross-validated log-likelihood) give differing results
    - :question: How much do CST and CO overlap in the neural manifold?
        - :bar_chart: compare neural covariance dimensions between CO and CST, using subspace overlap or principle angles
- Subspace analysis
    - :question: Is there a neural "go" signal underlying intermittent movements/control?
        - :bar_chart: Condition independent signal analysis
            - :speech_balloon: It's difficult to pull out a meaningful CIS at a single trial level, even for CO task. We probably need a method for trial-averaging CST to make use of this (like submovement decomposition)
    - :question: is there a dimension of neural activity corresponding to neural engagement? Perhaps something that corresponds to whether $\lambda$ changed on the last trial, or whether there have been previous failures?
        - :bar_chart: Analyze major neural dimensions during hold period after failures and $\lambda$ changes
    - :bar_chart: Behaviorally potent/null space analysis
        - :question: Do CO and CST share a kinematic potent space? (use principle angles and subspace alignment)
            - :heavy_check_mark: Sudarshan's and my own analysis suggests that potent spaces are at least partially aligned
        - :question: How much neural variance lies in the potent and null spaces of each task, compared to total neural variance?
        - :question: Is there any correspondence between the timing of null space activation and movement?
            - :speech_balloon: Note: our arrays don't seem to have preparatory activity, which is what you'd expect to find in the null space. But there may be some information about visual feedback, if we can isolate feedback integration times
        - :question: What does the null space activity look like during BCI CST, where potent space is specified?
            - :speech_balloon: Aaron says that BCI data might not be great? Need to check with Emily
    - :question: Are there dimensions of neural activity that separate CO and CST?
        - :heavy_check_mark: Hold time analysis says possibly in the very early part of the movement, but perhaps this is due to kinematic differences?
        - :question: if there's a task context dimension, how aligned with kinematic potent space is it? Perhaps this dimension separates different motor cortical dynamic regimes?
- Neural [dynamics](tags/dynamics)
    - :bar_chart: [Tangling analysis](tags/tangling)
        - :question: CST is much more feedback driven than CO. Does this show up as M1 dynamics being less apparently autonomous? That is, is neural tangling higher in CST than CO?
            - :heavy_check_mark: Tangling doesn't really appear to be much higher in CST than CO, suggesting that the feedback may be "predictable" most of the time. 
            - :speech_balloon: Does tangling depend on the amount of data you use to check it? Right now I'm thinking it probably does, but I can't be sure without simulation
        - :question: Does the timecourse of tangling over a CST or CO trial reveal anything about corrective movements? Perhaps tangling increases before corrective movements or after apparent errors to signify introduction of unexpected feedback.
            - :speech_balloon: May need to develop corrective movement detection to enable trial averaging, as well as apparent error detection (for CST, this might equate to moving the hand in the wrong direction)
    - :bar_chart: Neural dynamic modeling
        - :question: Does CST contain different neural dynamical regimes? e.g. [switching dynamics](tags/switching) between movement vs. hold/preparation
            - :exclamation: CO seems to have different regimes, according to previous work, but CST data seems a bit too noisy as is
        - :question: Does CST have autonomous and input-driven times in neural dynamics? That is, if we infer inputs to a neural dynamical system (AutoLFADS), do the inputs look similar between movement initiation and submovements?
            - :speech_balloon: Other work suggests movement initiation and corrective movements share features like condition independent signal and inferred inputs

## Behavioral analyses

- Inferrential
    - :question: We see an apparent "act and wait" type behavior in monkeys performing CST--they often don't respond to small cursor velocities. Can we decompose a subject's behavior into a set of submovements?
        - :bar_chart: Simple act and wait hidden markov model
        - :bar_chart: Switching controller strategy
- Generative
    - :question: What normative principles are necessary for a generative model to recapitulate the behavior of a subject (human or monkey) performing CST?
        - :bar_chart: Optimal feedback control (Mohsen)
        - :bar_chart: Reinforcement learning (Joel)

## Cross-over analyses

List of sophisticated behavioral analyses that could be run on neural data

- :bar_chart: Contraction analysis
- :question: Can we see a real difference between "position-controlled" or "velocity-controlled" trials according to OFC classification?
    - :bar_chart: Check neural averages between the two sets of trials
        - :exclamation: Note: need to control for different regimes of kinematics in the two sets of trials when comparing neural state across labels
    - :bar_chart: Use deep discriminator on behavior according to OFC-labeled trials (control: compare to random splits of trials)
