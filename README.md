# prime-gap-gpu - a new GPU program to find prime gaps.

A fast prime gap searching tool.

# Table of Contents

- [Overview](#overview)
- [Setup](#setup)

## Overview

TBD

## Setup

In general this is going to be easy under Ubuntu 24.04 or later

```bash
TODO trim
TODO cuda
$ sudo apt install libgmp10 libgmp-dev
$ sudo apt install mercurial build-essential automake autoconf bison make libtool texinfo m4
```

```
$ sudo apt install libmpfr-dev libmpc-dev libbenchmark-dev

$ python -m pip install --user gmpy2 primegapverify
```

```
$ git clone https://github.com/sethtroisi/prime-gap-gpu.git
$ cd prime-gap-gpu
```

## TODO

 * [ ] `save_partial_at_exit` save offset (and what merit this represents) + currently active M.
 * [ ] Might have missed '927598685 * 337# / 210 - 4538'?
