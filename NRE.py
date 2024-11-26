from NPE import NPEHandler
from logging import info, warning, error


class NREHandler(NPEHandler):
    def __init__(self, density_estimator,
                 density_estimator_kwargs: dict = {},
                 device = 'cuda'):
        super().__init__(density_estimator, density_estimator_kwargs, device)
        info("Succesfully initialized NREHandler")
        
    def sample(self, num_samples: int, x, sample_kwargs: dict = {}):
        self.density_estimator.build_posterior(sample_kwargs)
        return self.density_estimator.sample(num_samples=num_samples, x=x)