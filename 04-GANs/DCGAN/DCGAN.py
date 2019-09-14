import os,time,sys
import argparse
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Parameters
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to input folder")
ap.add_argument("-r", "--results", default='results', help="Path to results folder")
ap.add_argument("-e", "--epochs", type=int, default = 20, help="Training epochs")
ap.add_argument("-lr", "--learning_rate", type=float, default=0.0002, help="Learning rate")
ap.add_argument("-is", "--img_size", type=int, default = 64,help="Image size")
ap.add_argument("-bs", "--batch_size", type=int, default = 128, help="Batch size")
ap.add_argument("-nz", "--size_vector_z", type=int, default = 100, help="Size of z latent vector")
ap.add_argument("-nc", "--num_channels", type=int, default = 3, help="Number of channels")
ap.add_argument("-ngf", "--size_feature_g", type=int, default = 64, help="Size of feature maps in generator")
ap.add_argument("-ndf", "--size_feature_d", type=int, default = 64, help="Size of feature maps in discriminator")
ap.add_argument("-b", "--beta", type=float, default = 0.5, help="Beta 1: hyperparam for Adam optimizers")
ap.add_argument("-s", "--show", type=bool, default = True, help="Show progress for each epoch")
ap.add_argument("-ng", "--ngpu", type=int, default = 1, help="Available GPU(s)")
args = vars(ap.parse_args())

#Training parameters
batch_size = args["batch_size"]
lr = args["learning_rate"]
train_epoch = args["epochs"]
beta1 = args["beta"]

ngpu = args["ngpu"]
#Sizes
nz = args["size_vector_z"]
ngf = args["size_feature_g"]
ndf = args["size_feature_d"]
img_size = args["img_size"]
nc = args["num_channels"]

#Directories
data_dir = args["input"]
res_dir = args["results"]

if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
    sys.exit("Not a valid input folder")
    
dataset_name = os.path.basename(data_dir)

if not os.path.exists(res_dir):
    os.mkdir(res_dir)

if args["show"]:
    if not os.path.exists(os.path.join(res_dir,"Random_results")):
        os.mkdir(os.path.join(res_dir,"Random_results"))
    if not os.path.exists(os.path.join(res_dir,"Fixed_results")):
        os.mkdir(os.path.join(res_dir,"Fixed_results"))

# G(z)
class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0,bias=False),
                nn.BatchNorm2d(ngf*8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1,bias=False),
                nn.BatchNorm2d(ngf*4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1,bias=False),
                nn.BatchNorm2d(ngf*2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1,bias=False),
                nn.BatchNorm2d(ngf),
                 nn.ReLU(True),
                  # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Tanh()
            # state size. (nc) x 64 x 64
                )
    # forward method
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
             # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
             # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
             # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1,bias=False),
            nn.BatchNorm2d(ndf*8),
             nn.LeakyReLU(0.2, inplace=True),
             # state size. (ndf*8) x 4 x 4
             nn.Conv2d(ndf*8, 1, 4, 1, 0,bias=False),
               nn.Sigmoid()
               )
    # forward method
    def forward(self, input):
         return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))
    y1 = hist['D_losses']
    y2 = hist['G_losses']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()
        
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn(64, nz, 1, 1, device=device)
    G.eval()
    if isFix:
        test_images = G(fixed_noise)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Fixed Noise (Input to G)
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
      
# DCGAN Network
G = Generator().to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

# Load data
dataset = dset.ImageFolder(root=data_dir,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

# Train history
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

# Labels
real_label = 1
fake_label = 0

# Start time

start_time = time.time()

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == (round(train_epoch*0.5) + 1):
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("*********** Learning rate change! **************")

    if (epoch+1) == (round(train_epoch*0.75) + 1):
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("*********** Learning rate change! **************")

    num_iter = 0

    epoch_start_time = time.time()
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        D.zero_grad()
         # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
         # Forward pass real batch through D
        output = D(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = BCE_loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        
         ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = G(noise)
        label.fill_(fake_label)
         # Classify all fake batch with D
        output=D(fake.detach()).view(-1)
         # Calculate D's loss on the all-fake batch
        errD_fake = BCE_loss(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        D_optimizer.step()
        D_losses.append(errD.item())

         ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        G.zero_grad()
        # fake labels are real for generator cost
        label.fill_(real_label)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = D(fake).view(-1)
         # Calculate G's loss based on this output
        errG = BCE_loss(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        G_optimizer.step()

        G_losses.append(errG.item())

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d]\tTime: %.2f\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, train_epoch,per_epoch_ptime,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    if args["show"]:
        p = os.path.join(res_dir,'Random_results','{}_DCGAN_{}.png'.format(dataset_name,str(epoch + 1)))
        fixed_p = os.path.join(res_dir,'Fixed_results','{}_DCGAN_{}.png'.format(dataset_name,str(epoch + 1)))
        show_result((epoch+1), save=True, path=p, isFix=False)
        show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

torch.save(G.state_dict(), os.path.join(res_dir,"generator_param.pkl"))
torch.save(D.state_dict(), os.path.join(res_dir,"discriminator_param.pkl"))

show_train_hist(train_hist, save=True, path=os.path.join(res_dir,'DCGAN_train_hist.png'))

if args["show"]:
    images = []
    for e in range(train_epoch):
        img_name = os.path.join(res_dir,'Fixed_results','{}_DCGAN_{}.png'.format(dataset_name,str(e + 1)))
        images.append(imageio.imread(img_name))
    imageio.mimsave(os.path.join(res_dir,'generation_animation.gif'), images, fps=5)