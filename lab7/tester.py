import torch
import matplotlib.pyplot as plt

from lab7.AutoEncoder import Autoencoder
from loss import ModelLoss
from tqdm import tqdm


def plot_images(original=None, reconstructed=None, generated=None, num_images=5):
    if generated is not None:
        generated = generated[:num_images].cpu().permute(0, 2, 3, 1)  # [B, H, W, C]
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for i in range(num_images):
            axes[i].imshow(generated[i].clamp(0, 1))  # Clamp to [0, 1]
            axes[i].axis('off')
            axes[i].set_title("Generated")
        plt.tight_layout()
        plt.show()
    else:
        original = original[:num_images].cpu().permute(0, 2, 3, 1)  # [B, H, W, C]
        reconstructed = reconstructed[:num_images].cpu().permute(0, 2, 3, 1)
        fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
        for i in range(num_images):
            axes[0, i].imshow(original[i].clamp(0, 1))
            axes[0, i].axis('off')
            axes[0, i].set_title("Original")
            axes[1, i].imshow(reconstructed[i].clamp(0, 1))
            axes[1, i].axis('off')
            axes[1, i].set_title("Reconstructed")
        plt.tight_layout()
        plt.show()


def test_model(model, dataloader, model_type, checkpoint_path=None, device='cuda', plot_results=True, latent_dim=None):
    model = model.to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model weights from {checkpoint_path}")

    model.eval()
    loss_fn = ModelLoss(model_type=model_type)
    total_loss = 0

    with torch.no_grad():
        if model_type in ['autoencoder', 'vae']:
            for i, imgs in enumerate(tqdm(dataloader, desc="Testing")):
                imgs = imgs.to(device)

                if model_type == 'vae':
                    recon_imgs, mu, logvar = model(imgs)
                    loss = loss_fn(recon_imgs, imgs, mu, logvar)
                    total_loss += loss.item()

                    if plot_results and i == 0:
                        plot_images(original=imgs, reconstructed=recon_imgs)

                elif model_type == 'autoencoder':
                    recon_imgs = model(imgs)
                    loss = loss_fn(recon_imgs, imgs)
                    total_loss += loss.item()

                    if plot_results and i == 0:
                        plot_images(original=imgs, reconstructed=recon_imgs)

                else:
                    raise ValueError("Invalid model_type. Use 'autoencoder' or 'vae'.")

            avg_loss = total_loss / len(dataloader)
            print(f"Average {model_type.upper()} Test Loss: {avg_loss:.4f}")

        elif model_type == 'gan':
            if latent_dim is None:
                raise ValueError("latent_dim must be provided for GAN testing.")

            z = torch.randn(len(dataloader), latent_dim, device=device)
            generated_imgs = model(z)

            if plot_results:
                plot_images(generated=generated_imgs)
            print("Generated images plotted.")

        else:
            raise ValueError("Invalid model_type. Use 'autoencoder', 'vae', or 'gan'.")


if __name__ == '__main__':
    from VAE import VAE
    from dataloader import Dataset

    # Test VAE
    vae = VAE(latent_dim=512)
    test_loader = Dataset('./test.json')
    aue = Autoencoder()


    from Gan import Generator

    # Test GAN Generator
    generator = Generator(latent_dim=512)
    test_model(
        model=vae,
        dataloader=test_loader,
        model_type="vae",
        latent_dim=512,
        plot_results=True,checkpoint_path='./model_checkpoint_best_vae.pth',
        device='cuda'
    )
