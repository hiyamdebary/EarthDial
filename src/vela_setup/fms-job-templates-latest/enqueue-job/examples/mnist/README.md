# MNIST distributed training

To run the MNIST distributed training, modify the `mnist.yaml` filed with your `namespace` and `jobName`. Similarly, change the `volumes` list to provide a `persistentVolumeClaim` name that you may have access to (the values provided here are specific to the **development** FMS cluster in the `mcad-testing` namespace). Once the changes have been made run the following command:

```
$ helm upgrade --install --wait -f mnist.yaml mnist-training ../../chart
```
