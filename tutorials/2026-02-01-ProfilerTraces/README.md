I posted a video tutorial on this topic here:

https://youtu.be/vTdLpaI5gMQ

`base_train-profiled.py` contains the code that you can bolt on to a nanochat training run to save out the trace files.

* There's a block of code to set it up, line 400
* The profiler needs to be stepped, line 575
* Then a block of code to write out the trace files, line 629
