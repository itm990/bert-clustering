#!/bin/bash

# tokenization using juman

JUMAN=$HOME/.local/bin
JUMAN_MODEL=$HOME/.local/libexec/jumanpp/jumandic.jppmdl

# tokenization
sed 's/。/。\n/g' \
    | ${JUMAN}/jumanpp --model ${JUMAN_MODEL} -M \
    | sed 's/_[^ ]*//g' \
    | perl -pe 's/。 \n/。/g'

