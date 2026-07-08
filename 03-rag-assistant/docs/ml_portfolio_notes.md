# Notes on my ML portfolio (MLModel repo)

I built five projects in my MLModel repository on GitHub.

The first project was an iris species classifier using a Random Forest, which got 90% accuracy on the test set. The second was a regression model predicting diabetes disease progression, where Linear Regression (R² 0.45) narrowly beat Random Forest — a good lesson that simple baselines often win on small datasets.

The third project introduced MLOps with MLflow: I tracked three candidate classifiers (Logistic Regression, Random Forest, Gradient Boosting) and registered the winner, Logistic Regression at 96.67% accuracy, as "iris-classifier" version 1 in the model registry.

The fourth project served that model over REST using FastAPI, with automatic input validation and a /predict endpoint returning species plus confidence. The fifth wired up GitHub Actions CI so every push retrains all models and smoke-tests the API — my first green pipeline ran in 1 minute 37 seconds.

Key takeaway from the whole repo: training a model is the easy part; tracking, serving, and automating it is where engineering starts.
