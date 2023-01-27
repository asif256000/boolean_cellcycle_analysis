import multiprocessing as mp
import random
import time
import uuid

from utils import generate_categorical_hist


def test(x: int, y: int):
    return x * y, x + y


def generator_test(dummy_list: list, some_list: list[tuple]):
    for x, y in some_list:
        print(f"Sleeping for {(x+y)//2} seconds...")
        time.sleep((x + y) // 2)
        yield x + y, abs(x - y)
    time.sleep(0.2)
    print(f"Generator Complete. {dummy_list=}")


def some_test_fn(test_tup: tuple[int, int], dummy1: int = 0, dummy2: str = "null"):
    sum, diff = test_tup[0], test_tup[1]
    print(f"Sum: {sum}, Difference: {diff}")
    return sum, diff


def mp_test(some_tup: tuple[int, int]):
    print(f"First: {some_tup[0]}, Second: {some_tup[1]}")
    return some_tup[0], some_tup[1]


def queue_mp_handler(q: mp.Queue):
    args = q.get()
    return test(args[0], args[1])


def mp_handler(args: tuple):
    return test(args[0], args[1])


if __name__ == "__main__":
    # test_dict = dict()
    # for _ in range(1000):
    #     test_dict[uuid.uuid4().hex[:16]] = random.randrange(20)
    # generate_categorical_hist(test_dict, "test_plot", "test_title")

    dummy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    arguments = [
        (0, 1),
        (1, 1),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (2, 5),
        (3, 6),
        (7, 3),
        (1, 8),
        (4, 9),
        (5, 5),
        (6, 7),
    ]

    with mp.Pool(processes=4) as pool:
        print(list(pool.imap(some_test_fn, generator_test(dummy_list=dummy, some_list=arguments), chunksize=4)))

    # p = mp.pool.Pool(processes=4)
    # print(p.map(mp_handler, arguments))
    # q = mp.Queue(maxsize=4)

    # p = mp.Pool(4, initializer=queue_mp_handler, initargs=(q,))
    # for args in arguments:
    #     q.put(args)
    # for _ in range(4):
    #     q.put(None)
    # p.close()
    # p.join()
    # # while not q.empty():
    # #     print(q.get())
