# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def positional_encoding(x, num_frequencies=6, incl_input=True):
    results = [x] if incl_input else []
    for i in range(num_frequencies):
        freq = 2. ** i
        results.append(torch.sin(freq * np.pi * x))
        results.append(torch.cos(freq * np.pi * x))
    return torch.cat(results, dim=-1)

"""1.2 Complete the class model_2d() that will be used to fit the 2D image.

"""

class model_2d(nn.Module):
    """
    Define a 2D model comprising of three fully connected layers, two relu activations and one sigmoid activation.
    """
    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()

        ############################# TODO 1(b) BEGIN ############################
        # for autograder compliance, please follow the given naming for your layers
        self.layer_in = nn.Linear((num_frequencies*2+1)*2, filter_size)
        self.act1 = nn.ReLU()
        self.layer = nn.Linear(filter_size, filter_size)
        self.act2 = nn.ReLU()
        self.layer_out = nn.Linear(filter_size, 3)
        self.act_out = nn.Sigmoid()
        ############################# TODO 1(b) END #############################

    def forward(self, x):
        ############################# TODO 1(b) BEGIN ############################
        # example of forward through a layer: y = self.layer_in(x)
        x = self.layer_in(x)
        x = self.act1(x)
        x = self.layer(x)
        x = self.act2(x)
        x = self.layer_out(x)
        x = self.act_out(x)
        ############################# TODO 1(b) END #############################
        return x

def normalize_coord(height, width, num_frequencies=6):
    """
    Creates the 2D normalized coordinates, and applies positional encoding to them

    Args:
        height (int): Height of the image
        width (int): Width of the image
        num_frequencies (optional, int): The number of frequencies used in the positional encoding (default: 6).

    Returns:
        (torch.Tensor): Returns the 2D normalized coordinates (range in [0, 1]) after applying positional encoding to them.
                        Shape: [height*width, D*(2*num_frequencies+1)].
    """
    ############################# TODO 1(c) BEGIN ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them
    x = torch.linspace(0, 1, width)
    y = torch.linspace(0, 1, height)

    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    xy_grid = torch.stack((x_grid, y_grid), dim=-1).view(-1, 2).to(device)

    embedded_coordinates = positional_encoding(xy_grid, num_frequencies)
    ############################# TODO 1(c) END #############################

    return embedded_coordinates

"""You need to complete 1.1 and 1.2 first before completing the train_2d_model function. Don't forget to transfer the completed functions from 1.1 and 1.2 to the part1.py file and upload it to the autograder.

Fill the gaps in the train_2d_model() function to train the model to fit the 2D image.
"""

def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding, show=True):
    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000

    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
    model2d.apply(weights_init)

    # Define the optimizer
    optimizer = torch.optim.Adam(model2d.parameters(), lr=lr)

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []
    t = time.time()
    t0 = time.time()

    # Create the 2D normalized coordinates, and apply positional encoding to them
    embedded_coordinates = normalize_coord(height, width, num_frequencies)

    for i in range(iterations+1):
        optimizer.zero_grad()

        # Run one iteration
        # Compute mean-squared error between the predicted and target images. Backprop!
        pred = model2d(embedded_coordinates.to(device))
        pred = pred.reshape(height, width, 3)
        loss = torch.mean((pred - test_img)**2)
        loss.backward()
        optimizer.step()

        # Display images/plots/stats
        if i % display == 0 and show:
            # Calculate psnr
            psnr = 10 * torch.log10(1 / loss)

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(),
                  "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()
            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

    print('Done!')
    model_filename = 'model_2d_' + str(num_frequencies) + 'freq.pt'
    torch.save(model2d.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")
    plt.imsave('van_gogh_' + str(num_frequencies) + 'freq.png', pred.detach().cpu().numpy())
    return pred.detach().cpu()

"""Train the model to fit the given image without applying positional encoding to the input, and by applying positional encoding of two different frequencies to the input; L = 2 and L = 6."""

"""2.1 Complete the following function that calculates the rays that pass through all the pixels of an HxW image"""

def get_rays(height, width, intrinsics, w_R_c, w_T_c):
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
        height: the height of an image.
        width: the width of an image.
        intrinsics: camera intrinsics matrix of shape (3, 3).
        w_R_c: Rotation matrix of shape (3,3) from camera to world coordinates.
        w_T_c: Translation vector of shape (3,1) that transforms

    Returns:
        ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of each ray.
                                    Note that despite that all rays share the same origin, here we ask you to
                                    return the ray origin for each ray as (height, width, 3).
        ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the direction of each ray.
    """
    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)
    ray_origins = torch.zeros((height, width, 3), device=device)

    # Get intrinsics parameters
    f_x = intrinsics[0, 0]
    f_y = intrinsics[1, 1]
    c_x = intrinsics[0, 2]
    c_y = intrinsics[1, 2]

    # Create grid of pixel coordinates
    v, u = torch.meshgrid(torch.arange(0, width, device=device), torch.arange(0, height, device=device))

    # Normalize pixel coordinates
    x_normalized = (u - c_x) / f_x
    y_normalized = (v - c_y) / f_y

    # Stack normalized coordinates to form rays
    ray_directions[..., 0] = x_normalized
    ray_directions[..., 1] = y_normalized
    ray_directions[..., 2] = torch.ones_like(x_normalized)

    # Convert ray directions to world coordinates
    ray_directions = torch.matmul(ray_directions, w_R_c.transpose(0, 1))

    # Compute ray origins in world coordinates
    ray_origins = w_T_c.reshape(1, 1, 3).expand_as(ray_directions)

    return ray_origins, ray_directions

"""Complete the next function to visualize how is the dataset created. You will be able to see from which point of view each image has been captured for the 3D object. What we want to achieve here, is to being able to interpolate between these given views and synthesize new realistic views of the 3D object."""
# Load input images, poses, and intrinsics
data = np.load("lego_data.npz")

# Images
images = data["images"]

# Height and width of each image
height, width = images.shape[1:3]

# Camera extrinsics (poses)
poses = data["poses"]
poses = torch.from_numpy(poses).to(device)


# Camera intrinsics
intrinsics = data["intrinsics"]
intrinsics = torch.from_numpy(intrinsics).to(device)

# Hold one image out (for test).
test_image, test_pose = images[101], poses[101]
test_image = torch.from_numpy(test_image).to(device)

# Map images to device
images = torch.from_numpy(images[:100, ..., :3]).to(device)

plt.imshow(test_image.detach().cpu().numpy())
plt.show()


def plot_all_poses(poses):
    ############################# TODO 2.1 BEGIN ############################
    origins = []
    directions = []
    for pose in poses:
        # pose = torch.from_numpy(pose).to(device)
        w_R_c = pose[:3, :3]
        w_T_c = pose[:3, 3].unsqueeze(-1)
        ray_origins, ray_directions = get_rays(height, width, intrinsics.to(device), w_R_c, w_T_c)
        origins.append(ray_origins.cpu()[height//2, width//2])  # Append the origin of the ray passing through the center of the image
        directions.append(ray_directions.cpu()[height//2, width//2])  # Append the direction of the ray passing through the center of the image
    origins = torch.stack(origins)
    directions = torch.stack(directions)
    ############################# TODO 2.1 END #############################

    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(origins[..., 0].flatten(), origins[..., 1].flatten(), origins[..., 2].flatten(),
                  directions[..., 0].flatten(), directions[..., 1].flatten(), directions[..., 2].flatten(),
                  length=0.12, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.show()

plot_all_poses(poses)

"""2.2 Complete the following function to implement the sampling of points along a given ray."""

def stratified_sampling(ray_origins, ray_directions, near, far, samples):
    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
        ray_origins: Origin of each ray in the "bundle" as returned by the get_rays() function. Shape: (height, width, 3).
        ray_directions: Direction of each ray in the "bundle" as returned by the get_rays() function. Shape: (height, width, 3).
        near: The 'near' extent of the bounding volume.
        far: The 'far' extent of the bounding volume.
        samples: Number of samples to be drawn along each ray.

    Returns:
        ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
        depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """
    ############################# TODO 2.2 BEGIN ############################
    depth_points = torch.linspace(near, far, samples, device=ray_origins.device)
    depth_points = depth_points.expand(ray_origins.shape[0], ray_origins.shape[1], samples)

    ray_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_points[..., :, None]
    ############################# TODO 2.2 END #############################

    return ray_points, depth_points

"""2.3 Define the network architecture of NeRF along with a function that divided data into chunks to avoid memory leaks during training."""
class nerf_model(nn.Module):
    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        self.num_x_frequencies = num_x_frequencies
        self.num_d_frequencies = num_d_frequencies

        input_dim = (2 * (3 * num_x_frequencies) + 3)
        dir_dim = (2 * (3 * num_d_frequencies) + 3)

        # Define network layers compactly
        self.layers = nn.ModuleList([nn.Linear(input_dim, filter_size)] +
                                    [nn.Linear(filter_size, filter_size) for _ in range(4)] +
                                    [nn.Linear(filter_size + input_dim, filter_size)] +
                                    [nn.Linear(filter_size, filter_size) for _ in range(2)] +
                                    [nn.Linear(filter_size, 1), nn.Linear(filter_size, filter_size),
                                     nn.Linear(filter_size + dir_dim, filter_size // 2),
                                     nn.Linear(filter_size // 2, 3)])

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, d):
        pos_encoded = x
        dir_encoded = d

        # Pass through the initial layers with ReLU activation
        for i in range(5):
            x = self.relu(self.layers[i](x)) if i == 0 else self.relu(self.layers[i](x))

        # Skip connection concatenation
        x = torch.cat([x, pos_encoded], dim=-1)

        # Further processing through additional layers
        for i in range(5, 8):
            x = self.relu(self.layers[i](x))

        # Obtain the density (sigma)
        sigma = self.layers[8](x)

        # Process feature vector and concatenate with direction encoding
        x = self.layers[9](x)
        x = torch.cat([x, dir_encoded], dim=-1)
        x = self.relu(self.layers[10](x))

        # Get the RGB values
        rgb = self.sigmoid(self.layers[11](x))

        return rgb, sigma




def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    def get_chunks(inputs, chunksize=2**15):
        """
        This function gets an array/list as input and returns a list of chunks of the initial array/list
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    """
    This function returns chunks of the ray points and directions to avoid memory errors with the neural network.
    It also applies positional encoding to the input points and directions before dividing them into chunks,
    as well as normalizing and populating the directions.

    Return Shape:
        ray_points_batches: List of chunks of ray points. Each chunk has shape (H, W, S, 3*(2*num_x_frequencies+1)).
        ray_directions_batches: List of chunks of ray directions. Each chunk has shape (H, W, S, 3*(2*num_x_frequencies+1)).
    """
    ############################# TODO 2.3 BEGIN ############################
    # Apply positional encoding to the ray points and directions (you may normalize the directions here)
    # repeat the directions to match the dimension and number of points S
    H, W, S, _ = ray_points.shape
    ray_directions = ray_directions[..., None, :].expand(-1, -1, S, -1)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    ray_points_encoded = positional_encoding(ray_points.reshape(-1, 3), num_x_frequencies).reshape(H, W, S, -1)
    ray_directions_encoded = positional_encoding(ray_directions.reshape(-1, 3), num_d_frequencies).reshape(H, W, S, -1)

    # Divide the ray points and directions into chunks
    ray_points_batches = get_chunks(ray_points_encoded.reshape(-1, ray_points_encoded.shape[-1]))
    ray_directions_batches = get_chunks(ray_directions_encoded.reshape(-1, ray_directions_encoded.shape[-1]))
    ############################# TODO 2.3 END #############################

    return ray_points_batches, ray_directions_batches



"""2.4 Compute the compositing weights of samples on camera ray and then complete the volumetric rendering procedure to reconstruct a whole RGB image from the sampled points and the outputs of the neural network."""

def volumetric_rendering(rgb, s, depth_points):
    """
    Differentiably renders a radiance field, given the origin of each ray in the "bundle",
    and the sampled depth values along them.

    Args:
        rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
        s: Volume density sigma at each query location (X, Y, Z). Shape: (height, width, samples).
        depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
        rec_image: The reconstructed image after applying the volumetric rendering to every pixel. Shape: (height, width, 3)
    """
    ############################# TODO 2.4 BEGIN ############################
    H, W, N = s.shape

    # Compute the weights of each sample
    s = nn.functional.relu(s)
    delta = torch.zeros((H, W, N), device=device)
    delta[:, :, 0:-1] = depth_points[:, :, 1:] - depth_points[:, :, :-1]
    delta[:, :, -1] = 1e9
    alpha = 1. - torch.exp(-s * delta)

    T = torch.ones_like(alpha)
    T[:, :, 1:] = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :, :-1]

    weights = alpha * T

    # Compute the final color of each pixel
    rgb = torch.sum(weights[..., None] * rgb, dim=-2)

    # Normalize the color
    rec_image = rgb
    rec_image = rec_image.clamp(0, 1)
    ############################# TODO 2.4 END #############################

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    ############################# TODO 2.5 BEGIN ############################
    # Compute all the rays from the image
    w_R_c = pose[:3, :3]
    w_T_c = pose[:3, 3].unsqueeze(-1)
    ray_origins, ray_directions = get_rays(height, width, intrinsics, w_R_c, w_T_c)

    # Sample the points from the rays
    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    # Divide data into batches to avoid memory errors
    ray_points_batches, ray_directions_batches = get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)

    # Forward pass the batches and concatenate the outputs at the end
    rgb = None
    sigma = None
    for i in range(len(ray_points_batches)):
        rgb_batch, sigma_batch = model(ray_points_batches[i], ray_directions_batches[i])
        if rgb is None:
            rgb = rgb_batch
            sigma = sigma_batch
        else:
            rgb = torch.cat((rgb, rgb_batch), 0)
            sigma = torch.cat((sigma, sigma_batch), 0)

    rgb = rgb.reshape(height, width, samples, 3)
    sigma = sigma.reshape(height, width, samples)

    # Apply volumetric rendering to obtain the reconstructed image
    rec_image = volumetric_rendering(rgb, sigma, depth_points)
    ############################# TODO 2.5 END #############################

    return rec_image
