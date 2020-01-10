## Unbiased Pairwise Learning from Implicit Feedback

---

### About

This repository accompanies the real-world experiments conducted in the paper "Unbiased Pairwise Learning from Implicit Feedback".

<!--
which has been accepted to []().

If you find this code useful in your research then please cite:
```
@
```
-->

### Dependencies

- numpy==1.16.2
- pandas==0.24.2
- scikit-learn==0.20.3
- tensorflow==1.14.0
- plotly==3.10.0
- optuna==0.19.0
- mlflow==1.3.0

### Running the code

To run the experiment, download the Yahoo! R3 dataset from (https://webscope.sandbox.yahoo.com/catalog.php?datatype=r) and put `train.txt` and `test.txt` files into `data/yahoo/` directory. Then, run the below command in `src` directory

```
$ sh run.sh
```

This will run the main experiments with the Yahoo data reported in Section 5 of the paper.

You can see the default experimental settings in the `run.sh` file.
The tuned hyper-parameters can be found in `logs/*/yahoo/tuning` directory.
These default experimental parameters are actually used in our experiments.

Once the code is finished executing, you can view the run's metrics and parameters by running the command

```
$ mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

The experimental results can also be found in the `logs/*/yahoo/results/` directory.
