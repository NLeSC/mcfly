#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -e $1/bin/activate ]; then
	source $1/bin/activate;
fi

python -m noodles.worker ${@:2}

if [ -z ${VIRTUAL_ENV+x} ]; then
	deactivate;
fi
