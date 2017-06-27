import random
def gen_dirichlet_random(params):
    sample = [random.gammavariate(a, 1) for a in params]
    sample = [v/sum(sample) for v in sample]
    return sample

if __name__ == '__main__':
    print gen_dirichlet_random([1,2,3,4])

