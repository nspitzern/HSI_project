from time import perf_counter

from algorithms import MMCA, WALUMI, WALUDI, LP, ISSC

from HyperBenchmark.hyper_data_loader.HyperDataLoader import HyperDataLoader


def main():
    algos = {
        'MMCA': MMCA,
        'WALUMI': WALUMI,
        'WALUDI': WALUDI,
        'ISSC': ISSC,
        'LP': LP
    }
    NUM_OF_CLASSES = 10
    loader = HyperDataLoader()
    labeled_data = loader.generate_vectors("PaviaU", (1, 1))
    X, y = labeled_data[0].image, labeled_data[0].lables

    for alg_name, alg_f in algos.items():
        start = perf_counter()
        if alg_name == 'MMCA':
            model = alg_f(n_classes=NUM_OF_CLASSES)
        else:
            model = alg_f(n_bands=NUM_OF_CLASSES)

        model.fit(X)

        if alg_name == 'MMCA':
            bands = model.predict(X, clusters=y, eps=0.3)
        else:
            bands = model.predict(X)

        end = perf_counter()
        print(f'Algo: {alg_name}. ETA: {round(end - start)} seconds')
        print(bands)

        print('\n' * 2)


if __name__ == '__main__':
    main()
