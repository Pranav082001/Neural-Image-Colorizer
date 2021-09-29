# Neural-Image-Colorizer

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/osanseviero/Neural_Image_Colorizer)

Image Colorization using Autoencoders [PyTorch]

Heroku - https://neural-image-colorizer.herokuapp.com/

Medium - https://medium.com/@pranav.kushare2001/colorize-your-black-and-white-photos-using-ai-4652a34e967

<h1>Output</h1>
Input B/w image vs Output Colored Image


![download](https://user-images.githubusercontent.com/66110778/117584112-80a46e00-b128-11eb-9c67-d7b5aee0951c.png)
![outptu](https://user-images.githubusercontent.com/66110778/117584120-8c903000-b128-11eb-8713-5b9b3eada6e9.png)

<h1>Model</h1>

![autoencoder_model](https://user-images.githubusercontent.com/66110778/117584124-9ca80f80-b128-11eb-863f-5a4ed1696d00.png)

1] used Resnet18 for Encoder part

2] In decoder part latent representation (extracted features) are being upsampled 



<h1>Future Scope</h1>

1] Their are n number of ways to imporve this model

2] we can use different encoders like resnet50, inceptionnet, Densenet ,Vggnet 

3] I have just trained the base model because of time constraint

4] Results will improve in greater extent if model is trained for suitable number of epochs and dataset
