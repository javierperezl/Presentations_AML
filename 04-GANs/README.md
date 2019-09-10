
## GANs Homework

### DCGAN

 1. Successfully train a generative model on a dataset of your choosing.
* We consider a training as successful if both the generator and discriminator loss stabilizes over time and if the generated samples are somewhat reasonable new samples of the dataset given a fixed input, avoiding mode collapse (generator giving the same output for any input).
* Describe how the initial training went with the default parameters and a subset of the dataset, compare it to your most successful training and its corresponding modifications and justifications. 
* (Bonus) Implement a suggested technique to improve DCGANs Training (e.g. Add noise to discriminator, label swapping, label  smoothing,etc.)

#### Good samples example
![Good samples](https://raw.githubusercontent.com/IBIO4615-2019/Presentations_AML/master/04-GANs/assets/good_example.png)

#### Mode Colapse example
![Bad samples](https://raw.githubusercontent.com/IBIO4615-2019/Presentations_AML/master/04-GANs/assets/mode_collapse.png)

#### Stable training example
![Good training](https://raw.githubusercontent.com/IBIO4615-2019/Presentations_AML/master/04-GANs/assets/good_hist.png)

* Help: We provided a preprocessing script for transforming any dataset to 64x64 images. A recommended dataset for DCGAN is CelebA


 2. Implement a training scheme in which the Generator and Discriminator “take turns” to update, report on how this new scheme affected the training process in comparison to the previous point.  Provide an explanation of the additions, the modified code and some generated examples.(e.g. Update the discriminator every 2 epochs, update the generator or discriminator after a given loss value.)
 
 
 3. Implement a condition version of the DCGAN, using the demo code as a basis, training to generate numbers with MNIST. Provide an explanation of the necessary additions, the modified code and some generated examples.
* Recommendation: Use One Hot encoding  for the labels, the label y must be an additional input to the Generator and Discriminator.

![cDCGAN](https://raw.githubusercontent.com/IBIO4615-2019/Presentations_AML/master/04-GANs/assets/Example-cDCGAN.png)

#### data_loader

    transform = transforms.Compose([
            transforms.Scale(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataloader = torch.utils.data.DataLoader(
        dset.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
   
   To obtain the labels when using the dataloader use for data, label in dataloader:

     for data, label in dataloader:



### UGATIT

 4. Training U-GAT-IT with the [selfie2anime dataset](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view?usp=sharing), determine how important each of the loss functions are to the overall training result, by changing the loss weights 𝜆 in the hyperparameters, and describe what perceptual effects do their modification have on the generated samples. (Include images for each test)

* Recommendation: Use 128x128 images for the training, you can use the preprocessing script from DCGAN for the preprocessing, this point will take time, consider solving it first and try running it with the Azure resources.

![UGATIT Results](https://raw.githubusercontent.com/IBIO4615-2019/Presentations_AML/master/04-GANs/assets/UGATIT-Sample.JPG)

