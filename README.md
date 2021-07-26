# Keras-MMoE

This repo contains the implementation of [Multi-gate Mixture-of-Experts](http://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-) model in TensorFlow Keras.

Here's the [video explanation](https://www.youtube.com/watch?v=Dweg47Tswxw) of the paper by the authors.

The repository includes:
- A Python 3.6 implementation of the model in TensorFlow with Keras
    - The code is also compatible with Python 2.7
- Example demo of running the model with the [census-income dataset from UCI](https://bit.ly/2wLWmAY)
    - This dataset is the same one in Section 6.3 of the paper

The code is documented and designed to be extended relatively easy. If you plan on using this in your work, please consider citing this repository (BibTeX is included below) and also the paper.

## Getting Started

### Requirements
- Python 3.6
- Other libraries such as TensorFlow and Scikit-learn listed in `requirements.txt`

### Installation
1. Clone the repository
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the example code
```
python census_income_demo.py
```

## Notes
- Due to ambiguity in the paper and time and resource constraints, we unfortunately can't reproduce the exact results in the paper

## Contributing
Contributions to this repository are welcome. Examples of things you can contribute:
- Performance improvements by re-writing the model in TensorFlow, PyTorch, or MXNet
- Improve the census income benchmark to be as close as the one from the paper
- Improve the synthetic benchmark to be as close as the one from the paper
- Training on other public datasets for different benchmarks
- Accuracy improvements
- Visualizations

## Citation
Use this BibTeX to cite the repository:
```
@misc{keras_mmoe_2018,
  title={Multi-gate Mixture-of-Experts model in Keras and TensorFlow},
  author={Deng, Alvin},
  year={2018},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/drawbridge/keras-mmoe}},
}
```

## Acknowledgments
The code is built upon the [work](https://github.com/eminorhan/mixture-of-experts) by Emin Orhan.
