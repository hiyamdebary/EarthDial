# Run HuggingFace Transformer Models

## Background
Thanks to HF's Trainer API, all HF Transformer models that use trainer API (which covers nearly all HF models) work out of the box 
with torch distributed training. This makes FMS naturally capable of running distributed HF models natively.


## Running
The easiest way to run a HF model would be leveraging [enqueue-job](../..).  

Assuming we want to run [GLUE task](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification#glue-tasks).  

Three simple steps are all we need:
1. Download the [template](../../README.md#user_fileyaml) and save it as `hf.yaml`.
2. Copy the HF script to `mainProgram` section:
```
mainProgram: 
    run_glue.py 
        --model_name_or_path bert-base-cased 
        --task_name $TASK_NAME 
        --do_train 
        --do_eval 
        --max_seq_length 128 
        --per_device_train_batch_size 32 
        --learning_rate 2e-5 
        --num_train_epochs 3 
        --output_dir /workspace/$TASK_NAME/
```
3. Add necessary setups (e.g. dependencies) in `setupCommands` section:
```
setupCommands:
    - pip install transformers==4.26.0 datasets==2.9.0 evaluate
    - git clone -b v4.25-release https://github.com/huggingface/transformers.git
    - cd transformers/examples/pytorch/text-classification/
    - export HF_HOME=/workspace/.cache/
    - export TASK_NAME=MNLI
```

Once these are done, you should have something similar to [this](hf.yaml).

And that's it! Now you can run HF's GLUE Tasks using the following command:
```
helm template -f hf.yaml ../../chart | tee my-appwrapper.yaml | oc create -f -
```