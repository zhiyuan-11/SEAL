import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from estimators import *
from tqdm import tqdm

# Visualization function
def visualize_reconstruction(test_inputs, outputs, output_dir='output'):

    outputs = outputs.cpu()
    test_inputs = test_inputs.cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    outputs = outputs * std + mean
    test_inputs = test_inputs * std + mean

    outputs = torch.clamp(outputs, 0, 1)
    test_inputs = torch.clamp(test_inputs, 0, 1)

    fig, axs = plt.subplots(2, 6, figsize=(15, 5))
    for i in range(6):
        axs[0, i].imshow(np.transpose(test_inputs[i].numpy(), (1, 2, 0)))
        axs[0, i].axis('off')
        axs[0, i].set_title('Original')
        axs[1, i].imshow(np.transpose(outputs[i].numpy(), (1, 2, 0)))
        axs[1, i].axis('off')
        axs[1, i].set_title('Reconstructed')
    plt.show()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, 'reconstruction.png'))
    plt.close(fig)

# Training function with model saving
def train_decoder(model, decoder, train_loader, test_loader, criterion, optimizer, num_epochs=20, device='cuda', output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for epoch in range(num_epochs):
        decoder.train()
        running_loss = 0.0
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model.forward_features(inputs)
                last_hidden_state = outputs# .last_hidden_state  # (batch, 197, 768)
                # outputs = model(inputs)
                # last_hidden_state = outputs.last_hidden_state  # (batch, 197, 768)
            outputs = decoder(last_hidden_state)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        decoder.eval()
        with torch.no_grad():
            for batch_idx, (test_inputs, _) in enumerate(test_loader):
                test_inputs = test_inputs.to(device)
                outputs = model.forward_features(inputs)
                last_hidden_state = outputs# .last_hidden_state  # (batch, 197, 768)
                # outputs = model(inputs)
                outputs = decoder(last_hidden_state)
                break  

        visualize_reconstruction(test_inputs, outputs, output_dir)

        save_path = os.path.join(output_dir, 'decoder_public.pth')
        torch.save(decoder.state_dict(), save_path)
        print(f"Decoder model saved to {save_path}")
############################################################################################################
    
def visualize_difference(test_inputs, reconstructed_imgs, epoch, output_dir='output', is_sanity_check=False):
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Move tensors to CPU and clamp values for visualization
    test_inputs = test_inputs.cpu()
    reconstructed_imgs = reconstructed_imgs.cpu()

    # Denormalize the images using ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    test_inputs = test_inputs * std + mean
    reconstructed_imgs = reconstructed_imgs * std + mean

    test_inputs = torch.clamp(test_inputs, 0, 1)
    reconstructed_imgs = torch.clamp(reconstructed_imgs, 0, 1)

    # Calculate differences
    differences = (reconstructed_imgs - test_inputs).abs()

    # Create plot
    num_images = min(5, test_inputs.size(0))  # Limit to 5 images for display
    fig, axes = plt.subplots(3, num_images, figsize=(15, 9))
    for i in range(num_images):
        # Original Image
        axes[0, i].imshow(test_inputs[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        # Reconstructed Image
        axes[1, i].imshow(reconstructed_imgs[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')

        # Difference Heatmap
        diff_img = differences[i].abs().sum(axis=0)
        diff_img = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min())
        axes[2, i].imshow(diff_img, cmap='viridis')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Difference Heatmap')

    plt.suptitle(f'Epoch {epoch + 1}' if not is_sanity_check else 'Sanity Check')

    # Save figure
    if is_sanity_check:
        file_name = 'sanity_check.png'
    else:
        file_name = f'epoch_{epoch + 1}.png'
    
    fig_path = os.path.join(output_dir, file_name)
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Visualization saved at: {fig_path}")

def cross_cov_regularization(cls_origin, delta):
    """
    cls_origin: Tensor of shape (B, D), where row i is z_i
    delta:      Tensor of shape (B, D), where row i is delta_i
    
    We form the matrix M with M[i, j] = z_i^T delta_j.
    Then we encourage M[i, j] ~ M[i, i] for all j.
    The loss is the mean absolute difference (L1) of off-diagonal elements
    from the diagonal in each row.
    """
    # M[i, j] = (z_i) dot (delta_j)
    M = torch.matmul(cls_origin, delta.T)  # shape (B, B)
    
    # The diagonal of M: M[i, i] = z_i^T delta_i
    diag_M = torch.diag(M)  # shape (B,)
    
    # Expand diag_M to match M's shape for easy subtraction
    diag_M_expanded = diag_M.unsqueeze(1).expand_as(M)  # shape (B, B)
    
    # We want to penalize (M[i, j] - M[i, i]) for all i, j
    # L1 difference as an example:
    reg_matrix = (M - diag_M_expanded).abs()
    
    # Optionally, you can exclude the diagonal terms themselves
    # since M[i, i] - M[i, i] = 0. But including them won't hurt
    # because they contribute zero anyway.
    
    # Final loss: mean over all i, j
    reg_loss = reg_matrix.mean()
    
    return reg_loss

def train_ptb_transformer(model, decoder, ptb_transformer, train_loader, test_loader, criterion, optimizer, num_epochs=20, method='shannon', device='cuda', output_dir='output'):

    # Sanity check: reconstruct and visualize before training
    print("Performing sanity check before training...")
    ptb_transformer.eval()  # Make sure perturbation transformer is in eval mode for sanity check
    with torch.no_grad():
        test_inputs, _ = next(iter(test_loader))  # Get a batch of test images
        test_inputs = test_inputs.to(device)
        # outputs = model.forward_features(test_inputs) # model(test_inputs)
        # last_hidden_state = outputs #.last_hidden_state
        outputs = model(test_inputs)
        last_hidden_state = outputs.last_hidden_state
        reconstructed_imgs = decoder(last_hidden_state)
    
    visualize_difference(test_inputs, reconstructed_imgs, epoch=0, output_dir=output_dir, is_sanity_check=True)
    visualize_reconstruction(test_inputs, reconstructed_imgs, output_dir)

    Joint_entropy_alpha = Alpha_Joint_Entropy(kappa=1.0)
    MI_alpha = Alpha_Mutual_Information(kappa=0.5)
    frob_entropy_d = Frob_Entropy(kernel_type='rbf')
    frob_entropy_p = Frob_Entropy(kernel_type='vMF')
    MMD = MMDLoss(kernel_type='vmf', kappa=1.0)
    CSdiv = CSDivergence(kernel_type='vmf', kappa=1.0)
    CSqmi = CSQMI(kernel_type='vmf', kappa=1.0)

    for epoch in range(num_epochs):
        ptb_transformer.train()
        decoder.train()
        running_loss_rec = 0.0
        running_loss_joint_entropy = 0.0
        running_frob_entropy_d = 0.0
        running_frob_entropy_p = 0.0
        # running_patch_entropy = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, _) in enumerate(train_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                with torch.no_grad():
                    # outputs = model.forward_features(inputs) # model(inputs)
                    # last_hidden_state = outputs # .last_hidden_state
                    outputs = model(inputs)
                    last_hidden_state = outputs.last_hidden_state

                perturbed_hidden_state = ptb_transformer(last_hidden_state)
                reconstructed_imgs = decoder(perturbed_hidden_state)

                loss_rec = criterion(reconstructed_imgs, inputs)
                cls_origin = last_hidden_state[:, 0, :]
                cls_perturb = perturbed_hidden_state[:, 0, :]
                patch_origin = last_hidden_state[:, 1:, :].mean(dim=1)      
                patch_perturb = perturbed_hidden_state[:, 1:, :].mean(dim=1)
                delta = cls_perturb - cls_origin

                orm_delta = torch.norm(delta, p=2, dim=-1).mean()
                joint_entropy = CSqmi(cls_origin, cls_perturb)
                patch_joint_entropy = Joint_entropy_alpha(patch_origin, patch_perturb)
                entropy_delta = frob_entropy_d(delta, cls_origin)
                # entropy_perturbed = frob_entropy_p(cls_perturb, None)

                loss = loss_rec  - joint_entropy  # - entropy_delta 
                #- patch_joint_entropy # - 0.2 * entropy_delta   # - entropy_delta # # - 0.5 * entropy_perturbed
                loss.backward()
                optimizer.step()

                # Accumulate losses for the epoch
                running_loss_rec += loss_rec.item() * inputs.size(0)
                running_loss_joint_entropy += joint_entropy.item() * inputs.size(0)
                # running_frob_entropy_d += entropy_delta.item() * inputs.size(0)
                # running_frob_entropy_p += entropy_perturbed.item() * inputs.size(0)
                #running_patch_entropy += patch_joint_entropy.item() * inputs.size(0)

                pbar.update(1)
                pbar.set_postfix({
                    "Loss": f"{loss_rec.item():.4f}",
                    "Joint Entropy": f"{joint_entropy.item():.4f}"
                })

        # Compute the average metrics for the epoch
        epoch_loss_rec = running_loss_rec / len(train_loader.dataset)
        epoch_Joint_entropy = running_loss_joint_entropy / len(train_loader.dataset)
        # epoch_Frob_entropy_delta = running_frob_entropy_d / len(train_loader.dataset)
        # epoch_Frob_entropy_perturbed = running_frob_entropy_p / len(train_loader.dataset)
        # epoch_patch_entropy = running_patch_entropy / len(train_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Reconstruction Loss: {epoch_loss_rec:.4f}, "
              f"Joint Entropy: {epoch_Joint_entropy:.4f}, ")
              #f"Frob_Entropy_delta: {epoch_Frob_entropy_delta:.4f}, "
              #f"Frob_Entropy_perturbed: {epoch_Frob_entropy_perturbed:.4f}")
              # f"Patch Entropy: {epoch_patch_entropy:.4f}")
        
        # Visualize after each epoch
        ptb_transformer.eval()
        with torch.no_grad():
            for batch_idx, (test_inputs, _) in enumerate(test_loader):
                test_inputs = test_inputs.to(device)
                # outputs = model.forward_features(test_inputs) # model(test_inputs)
                # last_hidden_state = outputs #.last_hidden_state
                outputs = model(test_inputs)
                last_hidden_state = outputs.last_hidden_state
                perturbed_hidden_state = ptb_transformer(last_hidden_state)
                reconstructed_imgs = decoder(perturbed_hidden_state)
                break  

        visualize_difference(test_inputs, reconstructed_imgs, epoch=epoch, output_dir=output_dir, is_sanity_check=False)

        # Save the perturbation transformer model for each epoch
        save_path = os.path.join(output_dir, 'ptb_transformer_ablation.pth')
        torch.save(ptb_transformer.state_dict(), save_path)
        torch.save(decoder.state_dict(), os.path.join(output_dir, 'decoder_ablation.pth'))
        print(f"Perturbation transformer model for epoch {epoch+1} saved to {save_path}")