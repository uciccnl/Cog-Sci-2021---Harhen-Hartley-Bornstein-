# Code for "Model-based foraging using latent cause inference"

Harhen, Hartley, & Bornstein  (2021) Proceedings of the 43rd Annual Conference of the Cognitive Science Society.

## Key files

* Model.jl
  * The model described in the paper can be found in model.jl (function short_ref_point). To run the model, the subject number of the data you want given to the model must be specified, as well as the free parameters alpha (prior over cluster dispersion) and env_init (prior over environment richness).

* run_model.jl
  * Runs the model from the command line. The simulation generated behavior will be saved as a csv in the results folder.

* fit_model.jl
  * Runs the code to fit one participant using their behavior from four blocks (out of five). One blocks is left out as a test block for cross validation. Subject number and test block number are taken as arguments.

* load_fit.jl
  * Loads the best fitting participant for each participant-test block combination and simulates behavior with those fit parameters and saves results as a csv in the results folder.

* run_cross_val.jl
  * simulates behavior with fit parameters for both the multimodal (alpha_free) and unimodal (alpha_0) models and calculates the error between the predicted behavior (model simulation behavior) and the actual participant behavior on the  held-out test block. Uses this error to calculate a cross validation score and saves as a csv in results folder.

* example.ipynb
  * jupyter notebook demonstrating use of these files and plotting results.

For any questions, email nharhen@uci.edu!
