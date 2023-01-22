from multiprocessing import Process, pool


def test(x: int, y: int):
    return x * y


# p = pool.Pool(processes=4)
p = Process(target=test, args=[(0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (3, 4), (4, 5)])
p.start()
p.join()
