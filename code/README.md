To get the pretrained models, we must first combine the tar.gz.part\* archives (split due to Github file limits), so first run

cat models.tar.gz.parta\* >models.tar.gz

Next, we must unpack the archive with

tar xzf models.tar.gz

Finally, we can generate some samples with the following command:

bash sample\_model.sh

or 

bash sample\_markov\_model.sh

For more options in sampling such as temperature, manual specification of chord conditions, or sample chunk length, see the argparse help of each script that is called from the .sh wrappers
