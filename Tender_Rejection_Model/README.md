# ShipperGuide models

## Requirements

This notebook depends on:

- sklearn==1.3.0
- pandas==1.4.2
- holidays==0.13
- pricing-analysis-utils==0.15.1


## How to run

Follow the notebooks in the order they are numbered.

You need valid TRINO credentials before running the first notebook.

## Summary

In the following notebooks we prove the possibility of using simple models
with simple features to predict if:

- The bid has the lowest price
- The bid will be booked or not

We also show that the models are able to generalize to new and future data.

## Next steps

1. Analyze the probability curve depending on how we will use the models
2. Test different and more complex models
3. Add more features, such as benchmark
