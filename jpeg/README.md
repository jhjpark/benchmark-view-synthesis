# JPEG

This folder contains the source code used to run the JPEG image encoding and decoding techniques in Python. These were taken from [this](https://github.com/ghallak/jpeg-python) repository. The code can be run as follows. To encode a JPEG image, you can call the following command. This will output a binary file that contains the encoding of the JPEG image.
```
python encoder.py images/tractor.jpeg output/tractor.out
```

Then, this command can be used to decode that binary file to retrieve the original image.
```
python decoder.py output/tractor.out   
```
We also included a decoder that simply uses the `simplejpeg` library in `library_decoder.py`.
