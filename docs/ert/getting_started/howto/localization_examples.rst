Localization with examples from Drogon
=======================================

The open source synthetic reservoir model, Drogon, is used to illustrate the effect of using global update without localization
adaptive or distance-based localization on updates of gaussian field parameters.
In the following, global update means running an update in ERT without any localization. The observations
can influence the update of any model parameters regardless of any physical relation between a model parameter
and an observation due to spurious correlation. Localization is a common phrase for methods that try to mitigate
the unwanted effect of spurious correlation.

The Drogon case applies the facies model called 'Adaptive Pluri-Gaussian Simulation' (APS) since this
method uses Gaussian random fields as updatable parameters in ERT.


Example from Drogon
--------------------

The Drogon case was run with ES-MDA using the august 2026 version of ERT and the Drogon model. Three different
runs with ERT using ES-MDA with 3 updates and 100 realizations are used, one without
localization and with adaptive and distance-based localization. Adaptive localization used default threshold
for correlation. The ensemble mean and standard deviation and the differences between the posterior
and prior mean and standard deviations of one of the Gaussian fieldso from Drogon is shown for one layer
in one of the zones. The field statistics (mean, stdev) is calculated within
the help grid used for the field parameter (Ertbox help grid).

The first plot shows ensemble mean for the prior and the three different update strategies (global, adaptive, distance).
The global update will change the field parameter more or less uniformly while localization will update the field parameter
more locally. Note also that ensemble standard deviation is reduced a lot when using global update while localization
ensure that the updated ensemble standard deviation is much less reduced and more localized.
The distance-based localization clearly shows that the update is located to area around observations.

.. image:: ensemble_mean.png

The second plot shows the differences between posterior and prior ensemble mean for the three different update strategies.

.. image:: diff_posterior_prior_ensemble_mean.png

The third plot shows the ensemble standard deviation for the prior and the three different update strategies.

.. image:: ensemble_stdev.png

The fourth plot shows the differences between posterior and prior ensemble standard deviation for the three different update strategies.

.. image:: diff_posterior_prior_ensemble_stdev.png
