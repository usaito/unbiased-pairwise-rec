## Unbiased Pairwise Learning from Biased Implicit Feedback

---

### About

This repository accompanies the real-world experiments conducted in the paper "**Unbiased Pairwise Learning from Biased Implicit Feedback**" by [Yuta Saito](https://usaito.github.io/), which has been accepted by [ICTIR'20](https://ictir2020.org/).

<!-- If you find this code useful in your research then please cite:

```
@
``` -->


### Dependencies

- python>=3.7
- numpy==1.18.1
- pandas==0.25.1
- scikit-learn==0.23.1
- tensorflow==1.15.2
- pyyaml==5.1.2

### Datasets
To run the simulation with real-world datasets, the following datasets need to be prepared as described below.

- download the [Yahoo! R3 dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r) and put `train.txt` and `test.txt` files into `./data/yahoo/raw/` directory.
- download the [Coat dataset](https://www.cs.cornell.edu/~schnabts/mnar/) and put `train.ascii` and `test.ascii` files into `./data/coat/raw/` directory.

### Running the code

First, to preprocess the datasets, navigate to the `src/` directory and run the command

```bash
python preprocess_datasets.py -d coat yahoo
```

Then, run the following command in the same directory

```bash
for data in yahoo coat
  do
  for model in wmf expomf crmf bpr ubpr
  do
    python main.py -m $model -d $data -r 10
  done
done
```

This will run real-world experiments conducted in Section 4.
After running the experimens, you can summarize the results by running the following command in the `src/` directory.

```bash
python summarize_results.py -d yahoo coat
```

Once the code is finished executing, you can find the summarized results in `./paper_results/` directory.


### Acknowledgement

We thank [Minato Sato](https://github.com/satopirka) for his helpful comments, discussions, and advice.

