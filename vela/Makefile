.PHONY: build run clean

# Note: use vim to edit this file so \t characters are preserved (not the case in VSCode?)

DESC="GeoVLM Model"

PROJECT=eofm-base-geovlm-granite

build:
	docker build -t $(PROJECT) .

run:
	docker run -t $(PROJECT)

tag-remote:
	@echo 'tagging remote'
	docker tag $(PROJECT) us.icr.io/cil15-shared-registry/eofm/$(PROJECT):latest

push-remote:
	@echo 'push remote'
	docker push us.icr.io/cil15-shared-registry/eofm/$(PROJECT):latest