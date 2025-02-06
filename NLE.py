import torch
from NPE import NPEHandler
from logging import info, warning, error




class NLEHandler(NPEHandler):
    def __init__(self, density_estimator,
                 density_estimator_kwargs: dict = {},
                 device = 'cuda'):
        super().__init__(density_estimator, density_estimator_kwargs, device)
        self.posterior_constructed = False
        info("Succesfully initialized NLEHandler")
    
    def sample(self, num_samples: int, x, sample_kwargs: dict = {}):
        if not self.posterior_constructed:
            self.density_estimator.build_posterior(sample_kwargs)
            self.posterior_constructed = True
        return self.density_estimator.sample(num_samples=num_samples, x=x)

    def loss(self, img, lab, rnge):              
        # computing loss
        loss = self.density_estimator.loss(img, cond=lab)
        loss = loss.mean(0)

        return loss
    
    