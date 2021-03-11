from numpy.lib.utils import deprecate
import torch
import torch.nn as nn
import math
import numpy as np

SCALINGS = ["he", "mean", "none"]


class BinaryWeightMemory(object):
    r"""
    The goal of this class is to hold the original weights of the NN
    while we make use of their binarized version in a fault memory setting
    :param p: the proportion of bits that will not get randomly switched (p(x_ = x) = p)
    """

    def __init__(
        self,
        model: nn.Module,
        p: float = 0,
        scaling: str = "he",
        shortcutavoid: bool = False,
        skipfirst: bool = False,
    ):
        r"""
        Hold the pointer to the weights and the quantized representation associated
        From Courbariaux & al. 2015
        """
        ###
        # Input checks
        ###
        assert scaling in SCALINGS
        assert p > 0 and p <= 1, "P={} is not a probability (0<p<=1)".format(p)

        ###
        # Properties
        ###
        self.saved_params = []
        self.actual_params = []
        self.mask_faulty = []
        self.params, self.maxcons = 0, 0
        self.pis = []
        self.p = p
        self.layer_cardinality = []
        self.observed_fault_rate = []
        self.scaling = scaling

        ###
        # Registering weights
        ###
        skipped = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if not skipfirst or skipfirst and not skipped == 0:
                    self.saved_params.append(m.weight.data.clone())
                    self.actual_params.append(m.weight)
                    self.pis.append(p)
                    npar = sum(p.numel() for p in m.parameters() if p.requires_grad)
                    if (
                        hasattr(m, "name")
                        and m.name == "convShortcut"
                        and shortcutavoid
                    ):
                        self.pis[-1] = 0.5
                    self.layer_cardinality.append(npar)
                    self.observed_fault_rate.append(0.0)
                    self.mask_faulty.append(0.0)
                    self.maxcons += npar
                    self.params += 1
                else:
                    skipped += 1

        # print('***' * 10)
        # print(
        #     f'Initialising Binary Memory with state space {self.pis} (length {len(self.pis)})')
        # print("Layers cardinality")
        # print(self.layer_cardinality)
        # print(f'Consommation weights max {self.maxcons}')
        # print('***' * 10)
        assert (
            self.layer_cardinality.__len__()
            == self.saved_params.__len__()
            == self.actual_params.__len__()
            == self.params
        )

    def binarize(self):
        r"""
        Modify the weights to show their binary counterpart at inference time
        Reading is done in a simulated faulty memory setting : bits can get
        switched at random given a probability p of failure (mask tensor filled
        with 1 for good reading and -1 for failures)

        P(x_read=-1|x_real=1) = p
        P(x_read=1|x_real=-1) = p
        """
        # self.randomized = np.random.uniform(low=-self.random, high=self.random, size=self.params)
        # self.randomized = np.random.choice([-self.random, 0, self.random], self.params)#, p=[0.5,0.,0.5])
        for i in range(self.params):
            true_value = self.actual_params[i].data
            self.saved_params[i].copy_(self.actual_params[i].data)
            """ The mask is filled with 1 with 1-p probs
                then -1 with p probs to switch the sign.
                Element wise multiplication is then applied to compute the
                masked tensor (eg faulty memory tensor) """
            quantized = true_value.sign()
            mask_faulty = torch.rand_like(quantized).to(quantized.device)
            # TODO : try to grab directly in quantized the value rather than matmul (to test)
            # mask_faulty = torch.where(mask_faulty < sp[i], quantized, -quantized)
            mask_faulty = torch.where(
                mask_faulty >= self.pis[i],
                torch.tensor([1.0]).to(quantized.device),
                torch.tensor([-1.0]).to(quantized.device),
            )
            self.mask_faulty[i] = mask_faulty
            quantized *= mask_faulty
            if self.pis[i] == 0:
                assert torch.equal(quantized, true_value.sign())
            else:
                self.observed_fault_rate[i] = (
                    1
                    - (
                        torch.sum((mask_faulty + 1) / 2) / torch.numel(mask_faulty)
                    ).item()
                )

            if self.scaling == "he":
                quantized *= math.sqrt(2.0 / (np.prod(true_value.shape)))
                # quantized *= math.sqrt(
                # 2. / (true_value.shape[1] * true_value.shape[2] * true_value.shape[3]))
            elif self.scaling == "mean":
                quantized *= torch.mean(torch.abs(true_value))

            self.actual_params[i].data.copy_(quantized)

    def restore(self):
        for i in range(self.params):
            self.actual_params[i].data.copy_(self.saved_params[i])

    def clip(self):
        """From Courbariaux & al. 2015, 2.4 - Clip weights after update to -1;1
        since it doesn't impact the sign() ops while preventing overly large weights
        """
        for i in range(self.params):
            self.actual_params[i].data.copy_(
                torch.clamp(self.actual_params[i], min=-1, max=1).data
            )

    def __str__(self):
        """ Return a string representing the first param of the weight manager """
        return "Saved params \n {} \n Actual params \n {}".format(
            self.saved_params[0], self.actual_params[0]
        )

    @deprecate
    def change_p(self, new_p: float):
        """Allowed to change the global probability of bit switch
        _differentiate_ is now the method to change the dictionnary of probabilites per (bloc, layer) tuple
        """
        assert new_p >= 0 and new_p <= 1, "P={} is not a probability (0<p<=1)".format(
            new_p
        )
        self.pis = np.array(self.pis)
        self.pis.fill(new_p)

    def differentiate(self, new_state):
        """@param new_state : a ndarray which key correspond to the self.pis dictionnary
        Modify the value of self.pis
        """
        assert (
            new_state.__len__() == self.pis.__len__()
        )  # new_state should conform to the topology of the network
        new_state = np.clip(new_state, 0, 1)  # safety check
        self.pis = new_state
        print(f"Differentiate to {self.pis}")

    def get_state_space(self):
        """ Return self.pis represented as a ndarray """
        # if hasattr(self, 'randomized') and randomized:
        #     return self.pis + self.randomized
        # else:
        #     return self.pis
        return self.pis

    def get_consumption(self, p=None, constant=12.8):
        """Compute the energy function Σi η_i * Card(layer_i)
        Where η_i = - ln(pis)/12.8
        """
        if p is not None:
            pis = np.array(p)
        else:
            pis = np.array(self.pis)
            # if randomized:
            #     pis += np.array(self.randomized)
            #     pis = np.clip(pis, 0.5,1)
        # pis = 1-pis
        pis[pis == 0.5] = 1
        pis = -np.log(pis) / constant
        pis[pis == np.inf] = 1
        return np.sum(pis * self.layer_cardinality)

    def get_normalized_consumption(self, constant=12.8):
        """Consumption to 0-1 range
        self.get_consumption()/self.maxcons = avg(self.pis)
        """
        return self.get_consumption(constant=constant) / self.maxcons
