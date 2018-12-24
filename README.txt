In Project5 of Image Processing, I had to deal with the construction of neural
    networks (based on the ResNet architecture) for image restoration of:
    1. Image de-blurring.
    2. Image de-noising.

The Project has 3 main parts:

    Creating data -
     For each of the above, I took the appropriate data_set
     (images of text for de-blurring, regular for de-noising) and:
     1. Randomly divided it to 0.8 train set and 0.2 valid set.
     2. While training the model, I used a (python) generator that:
        A. Randomly selected a patch from a random image (with repetitions)
        C. Applied the corruption function for this patch
        D. Yield a tuple - (src_im, corrupted_im)

    Creating Models -
    I had to figure out what is the best num_res_blocks for our model, so I created
     5 networks (consisted of num_res_blocks from 1 to 5), and trained the 5 models.
    
    Evaluating Models -
    I compared val_err (on test_data, a data that I didn't use while training the net),
     and found that a network with 5 res_block minimized val_err.

I wrote this project in the first semester of my third year.