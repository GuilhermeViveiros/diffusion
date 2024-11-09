import torch
import numpy as np
from typing import List, Tuple
from constants import DEVICE


# the following class represents a linear scheduler that represents the amount of noise to be added in each step
class LinearScheduler:
    def __init__(self, n_steps: int, start: float = 0.0001, end: float = 0.02):
        # verify that the number of steps is valid
        assert n_steps > 0
        # verify that the start and end values are valid
        assert start >= 0 and start <= 1
        assert end >= 0 and end <= 1
        # verify that the start value is smaller than the end value
        assert start < end
        # save the number of steps
        self.n_steps = n_steps
        # save the start and end values
        self.start = start
        self.end = end
        # compute the step size
        self.step_size = (end - start) / n_steps
        # get linspace
        self.ts = torch.linspace(start, end, steps=n_steps, dtype=torch.float32)

    def step(self, t: int):
        # verify that the time step is valid
        assert t >= 0 and t < self.n_steps
        # return the noise value for the current time step
        return self.ts[t]


# the following class represents the forward diffusion process
class DiffusionProcess:
    def __init__(self, n_steps: int):
        # set nsteps
        self.n_steps = n_steps
        # get linear scheduler
        self.linear_scheduler = LinearScheduler(n_steps)
        # initialize variables
        self.__precompute_variables__()
        # print some information about diffusion process
        print("Diffusion Process:")
        print("\tnumber of steps: {}".format(self.linear_scheduler.n_steps))

    def __precompute_variables__(self):
        self._beta = torch.tensor(
            [
                self.linear_scheduler.step(t)
                for t in range(self.linear_scheduler.n_steps)
            ]
        )
        self._alpha = 1 - self._beta
        self._alpha_cumulative = torch.cumprod(self._alpha, dim=0)
        self._sqrt_alpha_cumulative = torch.sqrt(self._alpha_cumulative)
        self._one_by_sqrt_alpha = 1.0 / torch.sqrt(self._alpha)
        self._sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self._alpha_cumulative)

    # this function can diffuse/sample an image at any time step t
    def diffuse(self, x: torch.Tensor, t: int):
        # verify that the time step is valid
        assert t >= 0 and t < self.linear_scheduler.n_steps
        # compute the current productory noise value for time t
        alpha_cum = self._alpha_cumulative[t]
        # get noise from gaussian distribution
        noise = torch.rand_like(x)
        # get sample data
        xt = np.sqrt(alpha_cum) * x + np.sqrt(1 - alpha_cum) * noise
        # return the sample data
        return xt, noise

    # this function can diffuse/sample a list of tensor images at different time steps
    def vectorized_diffuse(
        self, xs: torch.Tensor, ts: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This function can diffuse/sample a list of tensor images at different time steps and return the diffused images and the noise
        Args:
            xs: tensor of images
            ts: list of time steps
        """
        # verify that the input tensor is valid.
        assert len(xs.shape) == 4, "xs should be a 4D tensor"
        # compute the current productory noise values for all time steps
        alpha_cums = torch.tensor([self._alpha_cumulative[t] for t in ts]).to(DEVICE)
        sqrt_one_minus_alpha_cumulative = torch.tensor(
            [self._sqrt_one_minus_alpha_cumulative[t] for t in ts]
        ).to(DEVICE)
        # get noise from gaussian distribution
        noise = torch.rand_like(xs)

        # get sample data
        xt = (torch.sqrt(alpha_cums.view(-1, 1, 1, 1)) * xs) + (
            sqrt_one_minus_alpha_cumulative.view(-1, 1, 1, 1) * noise
        )

        # return the sample data
        return xt, noise


# the following class represents the reverse diffusion process
# class ReverseDiffusionProcess():
#    def __init__(self, img_shape):


if __name__ == "__main__":
    # create a random image
    img = torch.rand(3, 256, 256)
    # test the diffusion process
    diffusion_process = DiffusionProcess(img, 1000)
    # diffuse image at time step 0
    xt = diffusion_process.diffuse(img, 0)
    # print information in a beautiful way
    print('Diffused Image:')
    print('\tshape: {}'.format(xt.shape))
    print('\tmin: {}, max: {}'.format(torch.min(xt), torch.max(xt)))
    # test the reverse diffusion process
