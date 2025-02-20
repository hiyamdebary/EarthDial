# Foundation model stack job launch using helm charts

## Assumptions

* The k8s/OpenShift target context (cluster) has MCAD, the co-scheduler, and Kubeflow training-operator installed. Cluster administrators can refer to the recommended list of operators [here](https://pages.github.ibm.com/ai-foundation/foundation-model-stack/current/install/required-platform-extensions/). 
* For using locally to launch jobs on the target k8s/OpenShift cluster, a cluster administrator has granted the user permissions to perform CRUD operations on `AppWrappers` in at least 1 namespace, and read operations for `PyTorchJob`, `PodGroup`, and `Pod` in the same namespace(s).

## Requirements for using locally

* The only utility needed to deploy the AppWrapped workloads is `helm` 3 (see installation instructions for your system [here](https://helm.sh/)).
* The OpenShift CLI ([`oc`](https://docs.openshift.com/container-platform/4.11/cli_reference/openshift_cli/getting-started-cli.html#installing-openshift-cli)) or k8s CLI (`kubectl`) are also handy (recommended).
* Login to the target k8s/OpenShift cluster.

## Workflow

### Create a `user_file.yaml`
Running distributed PyTorch training applications on OpenShift clusters requires setting up a number of cluster components such as number of workers, setup/running commands, volumes, etc. Our `helm` charts allow users to specify **only** the components that they care about leaving the remaining cluster setup for the utility to configure. To deploy a PyTorch distributed training job the user needs to provide a **user file** specifying the values and components they need for their job. An example of a user file is as follows:

<h5 a><strong><code>user_file.yaml</code></strong></h5>

``` yaml
# Job name and namespace
namespace: my-namespace
jobName: my-pytorch-job

# Container image to be used
#    Checkout https://github.com/foundation-model-stack/base-images/pkgs/container/base for other publicly available pytorch-nightly images
containerImage: ghcr.io/foundation-model-stack/base:pytorch-latest-nightly-20230126

# Runtime hardware specifications
numPods: 4
numCpusPerPod: 8
numGpusPerPod: 8
totalMemoryPerPod: 1Gi

# Any shell commands
setupCommands:
    - git clone https://github.com/dbarnett/python-helloworld
    - cd python-helloworld

# (For convenience) any single program to be passed to torchrun
mainProgram: helloworld.py
```

In the user file, users can specify the `namespace` for their deployment, the `jobName` to differentiate their jobs, and runtime specific requirements such as number of pods or/and the commands to run when the containers are ready.

**[All supported settings are described here](https://github.ibm.com/ai-foundation/foundation-model-stack/blob/main/tools/scripts/appwrapper-pytorchjob/chart/values.yaml)**, including things like environment variables, volumes, and other components which can be provided in the user file.

#### Recommended settings for training jobs on Vela, IBM’s AI-optimized, cloud-native supercomputer

**Vela upgraded with GDR over RoCE**

The following 2 steps are required only to users of the Vela cluster upgraded with GDR over RoCE.

1. One time per namespace aka `project`, run `oc apply -f fms-job-templates/configs/roce`, **assuming** your clone of this repository is your current working directory.

  Example
  ```
  bash-3.2$ pwd
  /Users/demodaemon/Documents/GitHub/fms-job-templates
  bash-3.2$ git pull
  Already up to date.
  bash-3.2$ oc project
  Using project "demo-tools" on server ...
  bash-3.2$ oc apply -f configs/roce
  configmap/nccl-netwk-env-vars created
  configmap/topo-gdr-2vf-canary created
  batch-3.2$ # done 
  ```

2. When launching jobs with `helm template` or `helm install` specify `-f ./enqueue-job/chart` **after** your `-f user_file.yaml`.

    Example (assumes current working directory is `fms-job-templates/enqueue-job/`):
    ```
    helm template -f user-file.yaml -f values-vela-roce.yaml ./chart | oc create -f -
    ```

During the transition, changes may be made to the `fms-job-templates/configs/roce/` configmaps installed in step (1) and to the additional values file `values-vela-roce.yaml` supplied in step (2). Please stay up-to-date with the repository, and with instructions from the cluster admins.


**More on Vela `user_file.yaml`**

An example user file is provided below, which shows recommended per pod resources (cpu, gpu, and memory), recommended environment variables, and the `multiNicNetworkName` for training workloads on Vela. These are good defaults to start fully leveraging the high throughput networking capabilities offered by Vela.
<h5 a><strong><code>user_file_vela.yaml</code></strong></h5>

``` yaml
# Job name and namespace
namespace: my-namespace
jobName: my-pytorch-job

# Container image to be used
#    Checkout https://github.com/foundation-model-stack/base-images/pkgs/container/base for other publicly available pytorch-nightly images
containerImage: ghcr.io/foundation-model-stack/base:pytorch-latest-nightly-20230126

# Runtime hardware specifications
numPods: 4
numCpusPerPod: 64
numGpusPerPod: 8
totalMemoryPerPod: 1Ti

# Commands
setupCommands:
    - git clone https://github.com/dbarnett/python-helloworld
    - cd python-helloworld

mainProgram: helloworld.py

# Expose multiple network interfaces to containers
multiNicNetworkName: multi-nic-network

# Type "nmon" inside the container and press 'n' to monitor network.
# During the training loop, you should see the network traffic is mainly through net1-0 and net1-1 interfaces.
environmentVariables:
    # Though usually NCCL is smart enough to pick faster network interfaces, no harm to enforce it via NCCL_SOCKET_IFNAME
    - name: NCCL_SOCKET_IFNAME
      value: "net1-0,net1-1"
    - name: NCCL_MIN_NCHANNELS
      value: "1"
    # Experiment with Tree when the message size is small
    - name: NCCL_ALGO
      value: Ring
    - name: NCCL_IGNORE_CPU_AFFINITY
      value: "1"
    - name: NCCL_IB_DISABLE
      value: "1"
    # Experiment with 4 when the message size is larger
    - name: NCCL_SOCKET_NTHREADS
      value: "2"
    # Experiment with 2 and see if it can further boost up the throughput
    - name: NCCL_NSOCKS_PERTHREAD
      value: "1"
    # Uncomment this when you need to debug NCCL runtime  
    #- name: NCCL_DEBUG_SUBSYS
    #  value: INIT,GRAPH,ENV,TUNING
    # Replace WARN to Info when you need to debug NCCL runtime
    - name: NCCL_DEBUG
      value: WARN
```

### Submit the job

Once a user file has been created and populated, submitting a distributed PyTorch application to the job queue is done by running the following command from this directory (if running from other directory, please provide the path to the `./chart` in the command bellow):

```
$ helm template -f ./user-file.yaml ./chart | tee my-appwrapper.yaml | oc create -f -
```

This saves the yaml to `my-appwrapper.yaml` and submits the job for validation and eventual launch. Example output:
```
appwrapper.mcad.ibm.com/my-appwrapper-test0 created
```

From this point, follow the [guide](https://pages.github.ibm.com/ai-foundation/foundation-model-stack/current/running/help-appwrapper/) to monitor and manage the job as an `AppWrapper` including eventual [clean-up](https://pages.github.ibm.com/ai-foundation/foundation-model-stack/current/running/help-appwrapper/#clean-up) e.g.

```
oc delete appwrapper.mcad.ibm.com/my-appwrapper-test0
```

## Advanced usage with helm release management

To launch using helm directly, use

```
$ helm install --wait -f path/to/user-file.yaml helm-deployment-name ./chart
```

Once finished with the job, clean-up with a single command:

```
$ helm uninstall --wait helm-deployment-name
```

An introduction to using helm can be found [here](https://helm.sh/docs/intro/using_helm/).

## Additional examples

Along with this utility, we provide a series of real world training examples that can be run in a given cluster. This set of examples will be expanded in future releases so please do come back here for future reference. To run the MNIST training example, follow the instructions in [here](examples/mnist).
