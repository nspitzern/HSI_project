from algorithms import MMCA

from HyperBenchmark.HyperDataLoader import HyperDataLoader


def main():
    loader = HyperDataLoader()
    pavia = loader.load_dataset_supervised("PaviaU")

    model = MMCA(n_bands=10)

    for i, (img, label) in enumerate(pavia):
        model.fit(img)
        bands = model.predict(img)
        print(bands)


if __name__ == '__main__':
    main()
