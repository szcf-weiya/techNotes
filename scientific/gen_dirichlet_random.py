import random
def gen_dirichlet_random_from_gamma(params):
    sample = [random.gammavariate(a, 1) for a in params]
    sample = [v/sum(sample) for v in sample]
    return sample

def gen_dirichlet_random_from_beta(params):
    xs = [random.betavariate(params[0], sum(params[1:]))]
    for j in range(1, len(params)-1):
        phi = random.betavariate(params[j], sum(params[j+1:]))
        xs.append((1-sum(xs)) * phi)
    xs.append(1-sum(xs))
    return xs

if __name__ == '__main__':
    print gen_dirichlet_random_from_gamma([1,2,3,4])
    print gen_dirichlet_random_from_beta([1,2,3,4])

