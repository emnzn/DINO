# momentum schedule = 0.99 - 1 from start to end epoch adjusted with cosing schedule
# weight decay schedule = 0.04 to 0.4 from start to end epoch adjusted with cosing schedule
# lr schedule = linear for 
    # first 10 epochs from 0 to 0.0005 * batchsize / 256
    # then from the base to 1.0e-6

# temperature schedule = 0.04 to 0.07 for first 30 epochs then constant at 0.07.
    # recommended to use constant 0.04 from repo.