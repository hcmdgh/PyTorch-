import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


class VAE(nn.Module):
    def __init__(self, c, h, w, latent_dim):
        super(VAE, self).__init__()

        self.c = c
        self.h = h
        self.w = w
        self.latent_dim = latent_dim

        # Encoder
        # [b, c, h, w] -> [b, 512, h/32, w/32]
        self.encoder = nn.Sequential(
            # [b, c, h, w] -> [b, 32, h/2, w/2]
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # [b, 32, h/2, w/2] -> [b, 64, h/4, w/4]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # [b, 64, h/4, w/4] -> [b, 128, h/8, w/8]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            # [b, 128, h/8, w/8] -> [b, 256, h/16, w/16]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # [b, 256, h/16, w/16] -> [b, 512, h/32, w/32]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        # Decoder
        # [b, 512, h/32, w/32] -> [b, 32, h/2, w/2]
        self.decoder = nn.Sequential(
            # [b, 512, h/32, w/32] -> [b, 256, h/16, w/16]
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # [b, 256, h/16, w/16] -> [b, 128, h/8, w/8]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            # [b, 128, h/8, w/8] -> [b, 64, h/4, w/4]
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # [b, 64, h/4, w/4] -> [b, 32, h/2, w/2]
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        # [b, 512, h/32, w/32] -> [b, h*w/2] -> [b, latent_dim]
        self.fc_mu = nn.Linear(h * w // 2, latent_dim)
        self.fc_sigma = nn.Linear(h * w // 2, latent_dim)

        # [b, latent_dim] -> [b, h*w/2]
        self.fc_decoder_in = nn.Linear(latent_dim, h * w // 2)

        # [b, 32, h/2, w/2] -> [b, c, h, w]
        self.decode_final_layer = nn.Sequential(
            # [b, 32, h/2, w/2] -> [b, 32, h, w]
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # [b, 32, h, w] -> [b, c, h, w]
            nn.Conv2d(in_channels=32, out_channels=c, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    # [b, c, h, w] -> [b, latent_dim]
    def encode(self, in_):
        # [b, c, h, w] -> [b, 512, h/32, w/32] -> [b, h*w/2]
        out = self.encoder(in_)
        out = torch.flatten(out, start_dim=1)

        # [b, h*w/2] -> [b, latent_dim]
        mu = self.fc_mu(out)
        log_sigma = self.fc_sigma(out)

        return mu, log_sigma

    # [b, latent_dim] -> [b, c, h, w]
    def decode(self, in_):
        # [b, latent_dim] -> [b, h*w/2]
        out = self.fc_decoder_in(in_)

        # [b, h*w/2] -> [b, 512, h/32, w/32]
        out = out.view(-1, 512, self.h // 32, self.w // 32)

        # [b, 512, h/32, w/32] -> [b, 32, h/2, w/2]
        out = self.decoder(out)

        # [b, 32, h/2, w/2] -> [b, c, h, w]
        out = self.decode_final_layer(out)

        return out

    # [b, latent_dim] -> [b, latent_dim]
    @staticmethod
    def reparameterize(mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, in_):
        mu, log_sigma = self.encode(in_)
        latent = self.reparameterize(mu, log_sigma)
        return latent, self.decode(latent), mu, log_sigma


def loss_function(real, recon, mu, log_sigma, kld_weight=0.005):
    # 重建损失?????
    recon_loss = F.mse_loss(recon, real)

    # KL散度?????
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim=1), dim=0)

    loss = recon_loss + kld_weight * kld_loss
    return loss


if __name__ == '__main__':
    img = torch.randn(1000, 3, 64, 64)
    model = VAE(c=3, h=64, w=64, latent_dim=10)
    torchsummary.summary(model, input_size=(3, 64, 64), device="cpu")
    out = model(img)
    print(out)
