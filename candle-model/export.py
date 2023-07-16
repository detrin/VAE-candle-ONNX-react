import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size=1024):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(size=h_dim),
            nn.ConvTranspose2d(h_dim, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


latent_size = 512
model = VAE(image_channels=3, h_dim=latent_size)
model.load_state_dict(torch.load("vae_candle_256.torch", map_location="cpu"))
model.eval()
example = torch.rand(1, 3, 256, 256)
traced_script_module = torch.jit.trace(model, example)

input_names = ["fc3_input0"]
output_names = ["fc3_output0"]
torch.onnx.export(
    model.fc3,
    torch.zeros((2)),
    "fc3.onnx",
    export_params=True,
    verbose=False,
    input_names=input_names,
    output_names=output_names,
    opset_version=12,
)

input_names = ["decoder_input0"]
output_names = ["decoder_output0"]
torch.onnx.export(
    model.decoder,
    torch.zeros((1, latent_size)),
    "decoder.onnx",
    export_params=True,
    verbose=False,
    input_names=input_names,
    output_names=output_names,
    opset_version=12,
)
