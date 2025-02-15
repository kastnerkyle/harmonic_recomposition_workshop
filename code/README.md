NOTE: FOR NOW ONLY SAMPLING FROM THE PRETRAINED MODELS IS SUPPORTED
-------------------------------------------------------------------
I still need to ensure the conversion from tfbldr to standalone lib is correct, but sampling pretrained models works now.

If you are interested in seeing the model + layer implementations right away, see any of the vq\_\*.py files, then you can explore lib to see the equivalent layer defs, which is a copy of https://github.com/kastnerkyle/tfbldr/tree/ea945ec8e3c4a797782da256c5c887713288032f

However, this is NOT a user facing library and no support of any kind will be availble. Feel free to copy, hack, and use in your own work though!


Required library dependencies for sampling
------------------------------------
numpy (any)

tensorflow (I used 1.4 and 1.5, but this should be flexible)

matplotlib (any)

scipy (any)

music21 (v .4 or .3)

pretty-midi (any)


Sampling Instructions
---------------------
To get the pretrained models, we must first combine the tar.gz.part\* archives (split due to Github file limits), so first run

cat models.tar.gz.parta\* >models.tar.gz

Next, we must unpack the archive with

tar xzf models.tar.gz

Finally, we can generate some samples with the following command:

bash sample\_model.sh

or 

bash sample\_markov\_model.sh

The resulting npz, mid, and png files will appear in a folder named samples/

If you wish to hear the midi files, use any midi player such as timidity (sudo apt-get install timidity in Ubuntu)

timidity myfile.mid

For more options in sampling such as temperature, manual specification of chord conditions, or sample chunk length, see the argparse help of each script that is called from the .sh wrappers
