from algorithms import MMCA

from HyperBenchmark.hyper_data_loader.HyperDataLoader import HyperDataLoader


def main():
    NUM_OF_CLASSES = 10
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables

    model = MMCA(n_classes=NUM_OF_CLASSES)

    model.fit(X)
    bands = model.predict(X, clusters=y)
    print(bands)


if __name__ == '__main__':
    main()
