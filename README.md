# README #

In a nutshell, this program time evolves the Schrodinger and Gross-Pitaevskii
equations in any number of required spatial dimensions (or, at least it will).

All of this (read as: most of this) takes place on CUDA enabled GPUs (OpenCL
would take much longer, but I do recognise the importance of that too).

So, you may ask, "How do I make it do stuff?".

Assuming you have the CUDA toolkit installed, and have enough GPU RAM to fit a
simulation on there, you just give it a few params and it will take care of the
rest. These can be specified in the text-file run_params.conf, or try the new
XML-based specifier (TBC).

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### Bullet points ###

* Install CUDA toolkit
* Install mencoder (RPMs/DEBs exist; or you can build it yourself)
* Install scipy, numpy, and matplotlib in python
* Come get some!

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
