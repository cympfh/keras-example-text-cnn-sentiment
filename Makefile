## training mode
train:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py train --name $(shell date "+%Y-%m-%d-%s")

## testing mode
test:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py test $(shell ls -1 snapshots/*.h5|peco)

## visplot a log
log:
		visplot --smoothing 2 -x epoch -y acc,val_acc $(shell ls -1 logs/*.json)

.DEFAULT_GOAL := help

## shows this
help:
	@grep -A1 '^## ' ${MAKEFILE_LIST} | grep -v '^--' |\
		sed 's/^## *//g; s/:$$//g' |\
		awk 'NR % 2 == 1 { PREV=$$0 } NR % 2 == 0 { printf "\033[32m%-18s\033[0m %s\n", $$0, PREV }'
