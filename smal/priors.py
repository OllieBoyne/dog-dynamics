import numpy as np
import os
import pickle as pkl
from chumpy import Ch
targ = r"E:\IIB Project Data\produced data\priors"

class Gaussian:
    """Multivariable gaussian distribution"""
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample_from(self):
        """Sample from dataset. Only works for unimodal currently"""
        return np.random.multivariate_normal(self.mean, self.cov)

    def __add__(self, other):
        assert isinstance(other, Gaussian), "Can only add two Gaussians"
        new = Gaussian
        new.mean = self.mean + other.mean
        new.cov = self.cov + other.cov
        return new#


    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            new = Gaussian(other * self.mean, other ** 2 * self.mean)

            return new

    def __rmul__(self, other):
        return self * other

class UnimodalPrior(Gaussian):
    options = ["unity_with_sf", "unity_without_sf", "random"]
    def __init__(self, name="unity_with_sf"):
        assert name in self.options, f"Name {name} not valid. Must be in {self.options}"

        if name != "random":
            self.name = name
            self.data = betas = np.load(os.path.join(targ, f"{name}.npy"))
            self.mean, self.cov = np.mean(betas, axis=0), np.cov(betas.T)

        else:
            self.mean = np.random.random((27))
            self.cov = np.random.random((27, 27))

        super().__init__(self.mean, self.cov)

pose_loc = r"E:\IIB Project Data\Dog 3D models\Pose Prior"
class PosePrior(Gaussian):
    options = ["zuffi", "05-05-20"]
    loc = pose_loc
    def __init__(self, name="zuffi"):
        with open(os.path.join(self.loc, name+".pkl"), "rb") as f:
            res = pkl.load(f, encoding="latin1")

        self.name = name
        self.mean = res['mean_pose']
        self.cov = res['cov']
        self.pic = res['pic'] # pic is the cholesky factorisation of the inverse of cov

        super().__init__(self.mean, self.cov)

    def as_dict(self, rots):
        """Given a set of rotations as [105], return as {j : [x,y,z]}, ignoring global rot"""
        rots = rots[3:] # ignore global rot
        return {j : rots[3*j : 3*j + 3] for j in range(34)}

    def get_mean(self):
        return self.as_dict(self.mean)

    def sample_from(self):
        """Return sample as dict of joint : [x, y, z]"""
        rots = np.random.multivariate_normal(self.mean, self.cov)
        return self.as_dict(rots)

class MixturePrior():
    options = ["mixture_1"]
    def __init__(self, name = "mixture_1"):
        assert name in self.options, f"Name {name} not valid. Must be in {self.options}"

        self.name = name
        data = np.load(os.path.join(targ, f"{name}.npz"))
        means, covar, coeff = data['means'], data['covar'], data['coeff']

        means[:, 20:] *=0
        n_clust, = coeff.shape
        self.priors = []
        for i in range(n_clust):
            prior = Gaussian(means[i], covar[i])
            self.priors.append(prior)


def pose_numpy_to_pkl(name, out_name):
    """Converts numpy array of (nx105) numpy pose fits to a pickle object of:
    'mean', 'cov', 'pic'"""
    raw = np.load(os.path.join(pose_loc, name))
    n_frames = raw.shape[0]

    # data = np.zeros((n_frames, 102))
    data = raw.reshape((n_frames, 102)) # get non global joint rots

    # data += np.random.random(data.shape)

    mean = np.mean(data, axis=0)
    cov = np.cov(data.T) + 1e-8 * np.eye(102) # add small amount to prevent matrix not being positive definite
    pic = np.linalg.cholesky(np.linalg.inv((cov)))
    # print(pic)

    # pre pad correctly to 'include' global rot
    out = {
        'mean_pose' : np.pad(mean, (3,0)),
        'cov' : np.pad(cov, ((3, 0), (3,0))),
        'pic' : Ch(np.pad(pic, ((3, 0), (3, 0)))),
    }

    with open(os.path.join(pose_loc, out_name), "wb") as outfile:
        pkl.dump(out, outfile)


# pose_numpy_to_pkl("05-05-20.npy", "unity_pose_prior_with_cov_35parts.pkl")

# u = UnimodalPrior()
# print(u.data[:, 21], u.data[:, 21].mean(), u.mean[21])