import torch
from logging import info, warning, error




class NPEHandler:
    def __init__(self, density_estimator,
                 density_estimator_kwargs: dict = {},
                 device = 'cuda'):
        self.density_estimator = density_estimator(**density_estimator_kwargs, device=device)
        self.device = device
        
        info("Succesfully initialized NPEHandler")
    
    def __call__(self, img, cond=None):
        return self.density_estimator.forward(img, cond)
    
    def to(self, device):
        self.density_estimator.to(device)
    
    def train(self):
        self.density_estimator.train()
        
    def eval(self):
        self.density_estimator.eval()
        
    def parameters(self):
        return self.density_estimator.parameters()
    
    def forward(self, img, cond=None):
        return self.density_estimator.forward(img, cond)
    
    def inverse(self, lab, cond=None):
        return self.density_estimator.inverse(lab, cond)
    
    def log_prob(self, lab, cond=None):
        return self.density_estimator.log_prob(lab, cond)
    
    def sample(self, num_samples: int, x, sample_kwargs: dict = None):
        return self.density_estimator.sample(num_samples=num_samples, x=x)
        
    def save(self, path: str = "./"):
        torch.save(self.density_estimator.state_dict(), path + "density_model.pt")
        
    def load(self, path: str = "./"):
        self.density_estimator.load_state_dict(torch.load(path + "density_model.pt", map_location=torch.device(self.device)))
        self.density_estimator.to(self.device)
        self.density_estimator.eval()

    def loss(self, img, lab, rnge):                  
        # computing loss
        loss = self.density_estimator.loss(lab, cond=img)
        loss = loss.mean(0)

        return loss
    
